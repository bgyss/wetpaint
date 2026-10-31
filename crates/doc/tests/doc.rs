use wet_compose::MixMode;
use wet_doc::Document;

fn blob(doc: &mut Document, cx: f32, color: [f32; 3]) {
    let w = doc.w;
    let l = doc.active_layer();
    for y in 20..40 {
        for x in 0..w {
            let d = (x as f32 - cx).abs();
            if d < 10.0 {
                l.deposit(y * w + x, 2.0, color, 300.0);
            }
        }
    }
}

#[test]
fn layers_composite_and_blend() {
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 20.0, [200.0, 40.0, 40.0]);
    doc.add_layer("top");
    blob(&mut doc, 30.0, [40.0, 40.0, 200.0]);
    assert_eq!(doc.layers.len(), 2);
    let normal = doc.render(None);
    doc.layers[1].mode = MixMode::Multiply;
    let multiply = doc.render(None);
    let px = |img: &[u8], x: usize| img[(30 * 64 + x) * 4..][..3].to_vec();
    // where the layers overlap, multiply is darker than normal
    let (n, m) = (px(&normal, 25), px(&multiply, 25));
    assert!(m.iter().sum::<u8>() as u32 <= n.iter().map(|&v| v as u32).sum::<u32>(), "{n:?} {m:?}");
    // a hidden layer disappears
    doc.layers[1].visible = false;
    assert_ne!(px(&doc.render(None), 35), px(&normal, 35));
}

#[test]
fn undo_redo_restore_layer_state() {
    let mut doc = Document::new(64, 64);
    let before = doc.render(None);
    doc.checkpoint();
    blob(&mut doc, 32.0, [10.0, 120.0, 30.0]);
    let after = doc.render(None);
    assert_ne!(before, after);
    assert!(doc.undo());
    assert_eq!(doc.render(None), before);
    assert!(doc.redo());
    assert_eq!(doc.render(None), after);
    assert!(!doc.redo());
}

#[test]
fn save_load_round_trip() {
    let mut doc = Document::new(48, 40);
    blob(&mut doc, 12.0, [200.0, 40.0, 40.0]);
    doc.add_layer("glaze");
    doc.layers[1].mode = MixMode::Screen;
    doc.layers[1].opacity = 0.5;
    blob(&mut doc, 30.0, [40.0, 200.0, 40.0]);
    for _ in 0..5 {
        doc.step();
    }
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    let mut back = Document::load(bytes.as_slice()).unwrap();
    assert_eq!(back.layers.len(), 2);
    assert_eq!(back.layers[1].name, "glaze");
    assert_eq!(back.layers[1].mode, MixMode::Screen);
    assert_eq!(back.render(None), doc.render(None));
    // the simulation continues identically after loading
    doc.step();
    back.step();
    assert_eq!(back.render(None), doc.render(None));
    assert!(Document::load(&b"nope"[..]).is_err());
}

#[test]
fn clipping_layer_mask_and_masking_fluid() {
    let mut doc = Document::new(64, 64);
    doc.paper.shading = 0.0; // flat paper so pixels can be compared
    blob(&mut doc, 20.0, [200.0, 40.0, 40.0]); // base: columns 10..30
    doc.add_layer("clipped");
    blob(&mut doc, 32.0, [40.0, 40.0, 200.0]); // columns 22..42
    doc.layers[1].clip = true;
    let img = doc.render(None);
    let px = |img: &[u8], x: usize| img[(30 * 64 + x) * 4..][..3].to_vec();
    // outside the base, the clipped layer is invisible: paper colour
    let paper = px(&img, 60);
    assert_eq!(px(&img, 38), paper);
    // a layer mask hides part of a layer
    doc.layers[1].clip = false;
    doc.layers[1].mask = Some((0..64 * 64).map(|i| if i % 64 < 32 { 255 } else { 0 }).collect());
    let img = doc.render(None);
    assert_eq!(px(&img, 38), paper);
    assert_ne!(px(&img, 24), paper);

    // masking fluid painted on its own layer protects the others and is not displayed
    let mut doc = Document::new(64, 64);
    doc.layers[0].masking_fluid = true;
    blob(&mut doc, 20.0, [255.0, 255.0, 255.0]);
    doc.add_layer("paint");
    doc.update_stencils();
    let w = doc.w;
    let l = doc.active_layer();
    for x in 0..w {
        l.deposit(30 * w + x, 1.0, [40.0, 40.0, 200.0], 300.0);
    }
    assert_eq!(doc.layers[1].wet.dens.amount.cur().data[30 * 64 + 20], 0.0);
    assert!(doc.layers[1].wet.dens.amount.cur().data[30 * 64 + 50] > 0.0);
    // round trip keeps the flags and the mask
    doc.layers[1].mask = Some(vec![128; 64 * 64]);
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    let back = Document::load(bytes.as_slice()).unwrap();
    assert!(back.layers[0].masking_fluid);
    assert_eq!(back.layers[1].mask.as_deref().map(|m| m[5]), Some(128));
}

#[test]
fn selection_limits_painting() {
    let mut doc = Document::new(64, 64);
    doc.select_shape((10.0, 10.0), (30.0, 30.0), false);
    doc.update_stencils();
    let w = doc.w;
    let l = doc.active_layer();
    for x in 0..w {
        l.deposit(20 * w + x, 1.0, [40.0, 40.0, 200.0], 300.0);
    }
    assert!(doc.layers[0].wet.dens.amount.cur().data[20 * 64 + 20] > 0.0);
    assert_eq!(doc.layers[0].wet.dens.amount.cur().data[20 * 64 + 40], 0.0);
    doc.selection = None;
    doc.update_stencils();
    assert!(doc.layers[0].wet.maps.stencil.is_none());
}

#[test]
fn transform_moves_and_scales_paint() {
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 20.0, [200.0, 40.0, 40.0]); // rows 20..40, cols 10..30
    let total = |d: &Document| d.layers[0].wet.dens.amount.cur().data.iter().sum::<f32>();
    let before = total(&doc);
    doc.transform_layer(0, (20.0, 0.0), 1.0, 0.0, (32.0, 32.0));
    let a = |d: &Document, x: usize, y: usize| d.layers[0].wet.dens.amount.cur().data[y * 64 + x];
    assert_eq!(a(&doc, 15, 30), 0.0);
    assert!(a(&doc, 45, 30) > 0.0);
    assert!((total(&doc) - before).abs() / before < 0.01);
    // colour survives the move
    let r = doc.layers[0].wet.dens.rgb[0].cur().data[30 * 64 + 45].to_f32();
    assert!((r - 200.0).abs() < 1.0, "{r}");
    // scaling ×2 about the paint centre roughly quadruples its area
    let mut d2 = Document::new(64, 64);
    blob(&mut d2, 32.0, [200.0, 40.0, 40.0]);
    let area = |d: &Document| d.layers[0].wet.dens.amount.cur().data.iter().filter(|&&v| v > 1.0).count();
    let a0 = area(&d2);
    d2.transform_layer(0, (0.0, 0.0), 2.0, 0.0, (32.0, 30.0));
    let a1 = area(&d2);
    assert!(a1 as f32 > 2.5 * a0 as f32, "{a0} -> {a1}");
}

#[test]
fn psd_export_structure() {
    let mut doc = Document::new(16, 48);
    blob(&mut doc, 5.0, [200.0, 40.0, 40.0]);
    doc.add_layer("top");
    doc.layers[1].mode = MixMode::Multiply;
    let mut bytes = Vec::new();
    doc.export_psd(&mut bytes).unwrap();
    if let Ok(p) = std::env::var("WETDOC_PSD_OUT") {
        std::fs::write(p, &bytes).unwrap();
    }
    assert_eq!(&bytes[..4], b"8BPS");
    assert_eq!(u16::from_be_bytes([bytes[4], bytes[5]]), 1);
    assert_eq!(u32::from_be_bytes(bytes[14..18].try_into().unwrap()), 48); // height
    assert_eq!(u32::from_be_bytes(bytes[18..22].try_into().unwrap()), 16); // width
    // layer count follows header (26) + colour data len (4) + resources len (4) + two lengths (8)
    let count = i16::from_be_bytes([bytes[42], bytes[43]]);
    assert_eq!(count, 2);
    assert!(bytes.windows(4).any(|w| w == b"mul "));
    // total size: merged image is 2 + 3·w·h bytes at the end
    let layer_section = u32::from_be_bytes(bytes[34..38].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 26 + 4 + 4 + 4 + layer_section + 2 + 3 * 16 * 48);
}

#[test]
fn lasso_wand_and_ops() {
    use wet_doc::SelectOp;
    let mut doc = Document::new(64, 64);
    // triangle lasso
    doc.select_polygon(&[(10.0, 10.0), (50.0, 10.0), (10.0, 50.0)], SelectOp::Replace);
    let sel = doc.selection.clone().unwrap();
    assert_eq!(sel[20 * 64 + 20], 255);
    assert_eq!(sel[45 * 64 + 45], 0);
    // subtract a rectangle-ish square polygon
    doc.select_polygon(&[(0.0, 0.0), (25.0, 0.0), (25.0, 64.0), (0.0, 64.0)], SelectOp::Subtract);
    let sel = doc.selection.clone().unwrap();
    assert_eq!(sel[20 * 64 + 15], 0);
    assert_eq!(sel[15 * 64 + 35], 255);
    // magic wand on a painted band: contiguous selects only the band
    doc.selection = None;
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    doc.select_color((32, 30), 30, true, SelectOp::Replace);
    let sel = doc.selection.clone().unwrap();
    assert_eq!(sel[30 * 64 + 32], 255);
    assert_eq!(sel[5 * 64 + 5], 0);
    doc.invert_selection();
    assert_eq!(doc.selection.as_ref().unwrap()[5 * 64 + 5], 255);
}

#[test]
fn groups_composite_isolated_and_round_trip() {
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 20.0, [200.0, 40.0, 40.0]);
    doc.add_layer("b");
    blob(&mut doc, 40.0, [40.0, 40.0, 200.0]);
    let plain = doc.render(None);
    let g = doc.group_layers("g", 0, 1);
    // a normal, opaque, visible group renders like its layers
    let grouped = doc.render(None);
    let diff = plain.iter().zip(&grouped).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(diff <= 1, "{diff}");
    // hiding the group hides both layers
    doc.groups[g].visible = false;
    let hidden = doc.render(None);
    assert_eq!(hidden[(30 * 64 + 20) * 4..][..3], hidden[(5 * 64 + 5) * 4..][..3]);
    // group opacity fades the members
    doc.groups[g].visible = true;
    doc.groups[g].opacity = 0.5;
    let faded = doc.render(None);
    let c = (30 * 64 + 20) * 4 + 1;
    assert!(faded[c] > grouped[c]);
    // .wetdoc round trip keeps groups
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    let back = Document::load(&bytes[..]).unwrap();
    assert_eq!(back.groups.len(), 1);
    assert_eq!(back.groups[0].opacity, 0.5);
    assert_eq!(back.layers[1].group, Some(0));
}

#[test]
fn psd_export_groups() {
    let mut doc = Document::new(16, 48);
    doc.add_layer("a");
    doc.add_layer("b");
    doc.group_layers("folder", 1, 2);
    let mut bytes = Vec::new();
    doc.export_psd(&mut bytes).unwrap();
    if let Ok(p) = std::env::var("WETDOC_PSD_GROUP_OUT") {
        std::fs::write(p, &bytes).unwrap();
    }
    // 3 layers + group start + group end
    assert_eq!(i16::from_be_bytes([bytes[42], bytes[43]]), 5);
    assert_eq!(bytes.windows(4).filter(|w| w == b"lsct").count(), 2);
}

#[test]
fn stencil_blocks_paint_and_wetting() {
    use wet_doc::Stencil;
    let mut doc = Document::new(64, 64);
    // a 16×16 opaque square stencil centred at (32, 30)
    doc.stencils.push(Stencil::new(vec![255; 256], 16, 16, (32.0, 30.0)));
    doc.update_stencils();
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    let l = &doc.layers[0].wet;
    let amount = |c: usize| l.mid.amount.cur().data[c] + l.dens.amount.cur().data[c];
    assert_eq!(amount(30 * 64 + 32), 0.0, "covered cell painted");
    assert!(amount(30 * 64 + 41) > 0.0, "uncovered cell not painted");
    // wetting the layer leaves the stencil area dry
    doc.layers[0].wet.wet_all(1.2);
    let l = &doc.layers[0].wet;
    assert_eq!(l.water.cur().data[25 * 64 + 32].to_f32(), 0.0);
    assert!(l.water.cur().data[5 * 64 + 5].to_f32() > 1.0);
    // inverted with a border: only the square is paintable
    doc.stencils[0].invert = true;
    doc.stencils[0].border = true;
    assert!(doc.stencils[0].coverage(32.0, 30.0) < 0.01);
    assert!(doc.stencils[0].coverage(5.0, 5.0) > 0.99);
    // tiled: the pattern repeats across the canvas
    let mut t = Stencil::new(vec![255, 0, 0, 255], 2, 2, (0.0, 0.0));
    t.tile = true;
    assert!((t.coverage(10.5, 10.5) - t.coverage(12.5, 12.5)).abs() < 1e-5);
}

#[test]
fn stencils_round_trip() {
    use wet_doc::Stencil;
    let mut doc = Document::new(32, 32);
    let mut st = Stencil::new((0..64).map(|i| (i * 4) as u8).collect(), 8, 8, (10.0, 12.0));
    st.rotation = 0.5;
    st.invert = true;
    doc.stencils.push(st);
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    let back = Document::load(&bytes[..]).unwrap();
    let (a, b) = (&doc.stencils[0], &back.stencils[0]);
    assert_eq!(a.mask, b.mask);
    assert_eq!((a.centre, a.rotation, a.invert), (b.centre, b.rotation, b.invert));
}

#[test]
fn layer_mask_apply_and_stencil_from_selection() {
    use wet_doc::SelectOp;
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    doc.select_polygon(&[(0.0, 0.0), (32.0, 0.0), (32.0, 64.0), (0.0, 64.0)], SelectOp::Replace);
    doc.add_layer_mask(0);
    let masked = doc.render(None);
    doc.apply_layer_mask(0);
    assert!(doc.layers[0].mask.is_none());
    let baked = doc.render(None);
    let diff = masked.iter().zip(&baked).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(diff <= 2, "{diff}");
    // paint on the masked-out side is gone
    let l = &doc.layers[0].wet;
    assert!(l.mid.amount.cur().data[30 * 64 + 38] + l.dens.amount.cur().data[30 * 64 + 38] < 1.0);
    // a stencil from the selection leaves the selected half paintable
    let st = doc.stencil_from_selection().unwrap();
    assert!(st.coverage(10.0, 10.0) < 0.01 && st.coverage(50.0, 10.0) > 0.99);
}

#[test]
fn cut_copy_and_select_visible() {
    use wet_doc::SelectOp;
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    let before = doc.render(None);
    doc.select_polygon(&[(0.0, 0.0), (32.0, 0.0), (32.0, 64.0), (0.0, 64.0)], SelectOp::Replace);
    let new = doc.copy_selection(0, true).unwrap();
    assert_eq!(new, 1);
    let amt = |doc: &Document, i: usize, c: usize| {
        let l = &doc.layers[i].wet;
        l.mid.amount.cur().data[c] + l.dens.amount.cur().data[c]
    };
    // the left half moved to the new layer, the right half stayed
    assert!(amt(&doc, 0, 30 * 64 + 26) < 1.0 && amt(&doc, 1, 30 * 64 + 26) > 10.0);
    assert!(amt(&doc, 0, 30 * 64 + 38) > 10.0 && amt(&doc, 1, 30 * 64 + 38) < 1.0);
    // the picture is unchanged (away from the anti-aliased selection edge, where two
    // partial coverages do not composite back to exactly one)
    let after = doc.render(None);
    let diff = (0..64 * 64)
        .filter(|c| !(30..=34).contains(&(c % 64)))
        .flat_map(|c| (0..3).map(move |k| 4 * c + k))
        .map(|i| before[i].abs_diff(after[i]))
        .max()
        .unwrap();
    assert!(diff <= 3, "{diff}");
    // select the visible pixels of the original layer: only its right half of the band
    doc.select_visible(0, SelectOp::Replace);
    let sel = doc.selection.as_ref().unwrap();
    assert!(sel[30 * 64 + 38] > 100 && sel[30 * 64 + 26] < 10 && sel[5 * 64 + 5] == 0);
}

#[test]
fn structure_raises_paper_relief() {
    use wet_doc::{Stencil, Structure};
    let mut doc = Document::new(64, 64);
    let flat = doc.render(None);
    let h0 = doc.layers[0].wet.maps.height.data[32 * 64 + 32];
    doc.structures.push(Structure { image: Stencil::new(vec![255; 100], 10, 10, (32.0, 32.0)), strength: 0.8 });
    doc.apply_structures();
    let h1 = doc.layers[0].wet.maps.height.data[32 * 64 + 32];
    assert!(h1 as i32 - h0 as i32 > 80, "{h0} -> {h1}");
    // the raised square's edges shade differently from the flat render
    let relief = doc.render(None);
    let edge = (32 * 64 + 27) * 4;
    assert_ne!(flat[edge], relief[edge]);
}

#[test]
fn deckled_paper_edges() {
    use wet_doc::PaperSettings;
    // small canvases (128 px) have no deckle, so use one above that
    let n = 160;
    let mut doc = Document::new(n, n);
    doc.set_paper(PaperSettings { deckled: true, ..PaperSettings::default() });
    let img = doc.render(None);
    assert_eq!(img[3], 0, "corner should be off the sheet");
    assert_eq!(img[(n / 2 * n + n / 2) * 4 + 3], 255);
    // painting beyond the edge is blocked
    for x in 0..n {
        doc.layers[0].wet.deposit(n / 2 * n + x, 1.0, [200.0, 40.0, 40.0], 300.0);
    }
    let l = &doc.layers[0].wet;
    assert_eq!(l.mid.amount.cur().data[n / 2 * n] + l.dens.amount.cur().data[n / 2 * n], 0.0);
    assert!(l.mid.amount.cur().data[n / 2 * n + n / 2] + l.dens.amount.cur().data[n / 2 * n + n / 2] > 0.0);
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    assert!(Document::load(&bytes[..]).unwrap().paper.deckled);
    // and none on a small canvas
    let mut small = Document::new(96, 96);
    small.set_paper(PaperSettings { deckled: true, ..PaperSettings::default() });
    assert_eq!(small.render(None)[3], 255);
}

#[test]
fn canvas_and_image_resize() {
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    let amt = |doc: &Document, x: usize, y: usize| {
        let l = &doc.layers[0].wet;
        let c = y * doc.w + x;
        l.mid.amount.cur().data[c] + l.dens.amount.cur().data[c]
    };
    let a0 = amt(&doc, 32, 30);
    // enlarge by 16 px on the left/top: content shifts
    doc.resize_canvas(96, 80, (16, 16));
    assert_eq!((doc.w, doc.h), (96, 80));
    assert!((amt(&doc, 48, 46) - a0).abs() < 1e-3);
    assert_eq!(amt(&doc, 48, 5), 0.0);
    assert_eq!(doc.render(None).len(), 96 * 80 * 4);
    // halve the image: the band is still there at half the position
    doc.resize_image(48, 40);
    assert!(amt(&doc, 24, 23) > 0.5 * a0);
}

#[test]
fn colour_profiles_tag_convert_and_embed() {
    use wet_color::RgbSpace;
    let mut doc = Document::new(32, 48);
    blob(&mut doc, 10.0, [200.0, 100.0, 50.0]);
    doc.convert_to_profile(RgbSpace::DisplayP3);
    let l = &doc.layers[0].wet;
    let c = 30 * 32 + 10;
    // sRGB (200, 100, 50) is (187, 105, 62) in Display P3
    let dens = [0, 1, 2].map(|k| l.dens.rgb[k].cur().data[c].to_f32().round());
    assert!((dens[0] - 187.0).abs() <= 2.0 && (dens[2] - 62.0).abs() <= 2.0, "{dens:?}");
    // PNG and PSD carry the profile
    let mut png = Vec::new();
    doc.export_png(&mut png).unwrap();
    if let Ok(dir) = std::env::var("WETDOC_ICC_OUT") {
        std::fs::write(format!("{dir}/p3.png"), &png).unwrap();
    }
    assert!(png.windows(4).any(|w| w == b"iCCP"));
    let mut psd = Vec::new();
    doc.export_psd(&mut psd).unwrap();
    if let Ok(dir) = std::env::var("WETDOC_ICC_OUT") {
        std::fs::write(format!("{dir}/p3.psd"), &psd).unwrap();
    }
    assert!(u32::from_be_bytes(psd[30..34].try_into().unwrap()) > 0);
    assert!(psd.windows(4).any(|w| w == b"acsp"));
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    assert_eq!(Document::load(&bytes[..]).unwrap().profile, Some(RgbSpace::DisplayP3));
}

#[test]
fn import_image_and_tracing_colour() {
    let mut doc = Document::new(32, 32);
    // left half blue, right half transparent
    let rgba: Vec<u8> = (0..32 * 32).flat_map(|i| if i % 32 < 16 { [30, 60, 200, 255] } else { [0, 0, 0, 0] }).collect();
    let idx = doc.import_image("ref", &rgba, 32, 32);
    let img = doc.render(None);
    let p = &img[(10 * 32 + 5) * 4..][..3];
    assert!(p[2] > 150 && p[0] < 80, "{p:?}");
    doc.layers[idx].tracing = true;
    let c = doc.tracing_color(5, 10).unwrap();
    assert!((c[0] - 30.0).abs() < 3.0 && (c[2] - 200.0).abs() < 3.0, "{c:?}");
    let paper = doc.tracing_color(25, 10).unwrap();
    assert!(paper.iter().all(|&v| v > 250.0), "{paper:?}");
}

#[test]
fn fill_tool_colour_gradient_and_options() {
    use wet_doc::{Fill, FillOptions};
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]); // a red band splits the paper
    // contiguous colour fill of the paper above the band stops at it
    doc.add_layer("fill");
    doc.fill((5, 5), &Fill::Color([40.0, 160.0, 60.0]), FillOptions::default());
    let img = doc.render(None);
    let px = |x: usize, y: usize| [img[(y * 64 + x) * 4], img[(y * 64 + x) * 4 + 1], img[(y * 64 + x) * 4 + 2]];
    assert!(px(5, 5)[1] > 120 && px(5, 5)[0] < 80, "{:?}", px(5, 5));
    assert!(px(32, 30)[0] > px(32, 30)[1], "band must stay red: {:?}", px(32, 30));
    // half-opacity gradient below the band, wet
    doc.fill(
        (5, 60),
        &Fill::Gradient { from: (0.0, 0.0), to: (63.0, 0.0), colors: [[0.0, 0.0, 200.0], [200.0, 200.0, 0.0]], radial: false },
        FillOptions { opacity: 0.5, wet: true, ..FillOptions::default() },
    );
    let l = &doc.layers[doc.active].wet;
    assert!(l.water.cur().data[60 * 64 + 5].to_f32() > 1.0);
    let img = doc.render(None);
    let (left, right) = (&img[(60 * 64 + 2) * 4..][..3], &img[(60 * 64 + 61) * 4..][..3]);
    assert!(left[2] > left[0] && right[0] > right[2], "{left:?} {right:?}");
}

#[test]
fn perspective_deform_and_warp() {
    let amt = |doc: &Document, x: usize, y: usize| {
        let l = &doc.layers[0].wet;
        l.mid.amount.cur().data[y * 64 + x] + l.dens.amount.cur().data[y * 64 + x]
    };
    let painted = || {
        let mut doc = Document::new(64, 64);
        blob(&mut doc, 32.0, [200.0, 40.0, 40.0]); // band x 22..42, y 20..40
        doc
    };
    // identity corners change nothing
    let mut doc = painted();
    let before = amt(&doc, 32, 30);
    doc.perspective_layer(0, [(0.0, 0.0), (63.0, 0.0), (63.0, 63.0), (0.0, 63.0)]);
    assert!((amt(&doc, 32, 30) - before).abs() < 1.0);
    // shrinking into the left half moves the band to x ≈ 16
    doc.perspective_layer(0, [(0.0, 0.0), (31.5, 0.0), (31.5, 63.0), (0.0, 63.0)]);
    assert!(amt(&doc, 16, 30) > 0.5 * before && amt(&doc, 40, 30) < 1.0);
    // deform with the same corners agrees (an affine quad)
    let mut d = painted();
    d.deform_layer(0, [(0.0, 0.0), (31.5, 0.0), (31.5, 63.0), (0.0, 63.0)]);
    assert!(amt(&d, 16, 30) > 0.5 * before && amt(&d, 40, 30) < 1.0);
    // a 1×1 warp grid shifting every point right by 10 moves the band by 10
    let mut wdoc = painted();
    wdoc.warp_layer(0, 1, &[(10.0, 0.0); 4]);
    assert!(amt(&wdoc, 46, 30) > 0.5 * before && amt(&wdoc, 24, 30) < 1.0);
}

#[test]
fn gloss_highlights_relief() {
    let mut doc = Document::new(64, 64);
    // a smooth dome of thick paint
    let l = &mut doc.layers[0].wet;
    for y in 12..52 {
        for x in 12..52 {
            let c = y * 64 + x;
            let r2 = ((x as f32 - 32.0).powi(2) + (y as f32 - 32.0).powi(2)) / 120.0;
            l.mid.amount.cur_mut().data[c] = 2000.0;
            l.mid.thickness.cur_mut().data[c] = 1200.0 * (-r2).exp();
            for k in 0..3 {
                l.mid.rgb[k].cur_mut().data[c] = half::f16::from_f32([150.0, 40.0, 40.0][k]);
            }
        }
    }
    doc.impasto = 1.0;
    let plain = doc.render(None);
    doc.gloss = 1.0;
    let glossy = doc.render(None);
    // the upper-left slope of the dome faces the light and catches a highlight
    let brighter = (0..64 * 64).filter(|&c| glossy[c * 4] as i32 > plain[c * 4] as i32 + 20).count();
    assert!(brighter > 10, "{brighter}");
    assert_eq!(plain[(5 * 64 + 5) * 4], glossy[(5 * 64 + 5) * 4], "bare paper has no gloss");
}

#[test]
fn pick_color_sources() {
    use wet_doc::PickSource;
    let mut doc = Document::new(64, 64);
    blob(&mut doc, 32.0, [200.0, 40.0, 40.0]);
    doc.add_layer("empty");
    // the active (empty) layer has nothing to pick; all layers find the red band
    assert!(doc.pick_color((32, 30), PickSource::CurrentLayer, 3).is_none());
    let all = doc.pick_color((32, 30), PickSource::AllLayers, 3).unwrap();
    assert!(all[0] > 150.0 && all[1] < 80.0, "{all:?}");
    // with the canvas, bare paper picks the paper colour
    let paper = doc.pick_color((5, 5), PickSource::AllLayersAndCanvas, 1).unwrap();
    assert!(paper.iter().all(|&v| v > 200.0), "{paper:?}");
    assert!(doc.pick_color((5, 5), PickSource::AllLayers, 1).is_none());
}

#[test]
fn pigment_layers_mix_subtractively() {
    // yellow layer under a blue glaze: additive-style compositing stays greyish, pigment
    // mixing gives green
    let make = |pigments: bool| {
        let mut doc = Document::new(64, 64);
        blob(&mut doc, 32.0, [240.0, 220.0, 30.0]);
        doc.add_layer("blue");
        blob(&mut doc, 32.0, [30.0, 60.0, 220.0]);
        doc.set_layer_pigments(1, pigments);
        doc.layers[1].opacity = 0.5;
        let img = doc.render(None);
        let p = &img[(30 * 64 + 32) * 4..][..3];
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };
    let (plain, pig) = (make(false), make(true));
    // green dominance (G over R and B) is stronger with pigments
    let green = |c: [f32; 3]| c[1] - 0.5 * (c[0] + c[2]);
    assert!(green(pig) > green(plain) + 5.0, "plain {plain:?} pigment {pig:?}");
    let mut doc = Document::new(16, 48);
    doc.set_layer_pigments(0, true);
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    assert!(Document::load(&bytes[..]).unwrap().layers[0].pigments);
}

#[test]
fn psd_import_round_trip() {
    let mut doc = Document::new(32, 48);
    blob(&mut doc, 10.0, [200.0, 40.0, 40.0]);
    doc.add_layer("top");
    doc.layers[1].mode = MixMode::Multiply;
    doc.layers[1].opacity = 0.5;
    blob(&mut doc, 22.0, [40.0, 60.0, 200.0]);
    doc.add_layer("a");
    doc.group_layers("g", 2, 2);
    let mut bytes = Vec::new();
    doc.export_psd(&mut bytes).unwrap();
    let mut other = Document::new(32, 48);
    assert_eq!(other.import_psd(&bytes), Some(3));
    let l = &other.layers[2]; // Layer 1 imported above the existing empty layer
    assert_eq!(l.name, "top");
    assert_eq!(l.mode, MixMode::Multiply);
    assert!((l.opacity - 0.5).abs() < 0.01);
    let before = doc.render(None);
    let after = other.render(None);
    let diff = before.iter().zip(&after).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
    assert!(diff <= 6, "{diff}");
}

#[test]
fn psd_reader_handles_rle() {
    // a 4×2 single-layer PSD with RLE channels, built by hand
    let (w, h) = (4usize, 2usize);
    let rle_row = |v: u8| vec![(-(w as i8 - 1)) as u8, v]; // repeat v w times
    let mut f: Vec<u8> = b"8BPS".to_vec();
    f.extend(1u16.to_be_bytes());
    f.extend([0; 6]);
    f.extend(3u16.to_be_bytes());
    f.extend((h as u32).to_be_bytes());
    f.extend((w as u32).to_be_bytes());
    f.extend(8u16.to_be_bytes());
    f.extend(3u16.to_be_bytes());
    f.extend(0u32.to_be_bytes());
    f.extend(0u32.to_be_bytes());
    let mut info: Vec<u8> = 1i16.to_be_bytes().to_vec();
    let chan_bytes = 2 + 2 * h + 2 * h; // compression + counts + 2 bytes per row
    info.extend(0u32.to_be_bytes());
    info.extend(0u32.to_be_bytes());
    info.extend((h as u32).to_be_bytes());
    info.extend((w as u32).to_be_bytes());
    info.extend(4u16.to_be_bytes());
    for id in [-1i16, 0, 1, 2] {
        info.extend(id.to_be_bytes());
        info.extend((chan_bytes as u32).to_be_bytes());
    }
    info.extend(b"8BIMnorm");
    info.extend([255, 0, 0, 0]);
    info.extend(12u32.to_be_bytes());
    info.extend(0u32.to_be_bytes());
    info.extend(0u32.to_be_bytes());
    info.extend([3, b'r', b'l', b'e']);
    for v in [255u8, 10, 20, 30] {
        info.extend(1u16.to_be_bytes());
        for _ in 0..h {
            info.extend(2u16.to_be_bytes());
        }
        for _ in 0..h {
            info.extend(rle_row(v));
        }
    }
    f.extend(((info.len() + 8) as u32).to_be_bytes());
    f.extend((info.len() as u32).to_be_bytes());
    f.extend(&info);
    f.extend(0u32.to_be_bytes());
    f.extend(0u16.to_be_bytes());
    f.extend(vec![128u8; 3 * w * h]);
    let psd = wet_doc::psd::read_psd(&f).unwrap();
    assert_eq!(psd.layers.len(), 1);
    assert_eq!(psd.layers[0].name, "rle");
    assert_eq!(&psd.layers[0].rgba[..4], &[10, 20, 30, 255]);
    assert_eq!(psd.composite[0], 128);
}

#[test]
fn psd_reader_external_file() {
    // optional: WETDOC_PSD_EXT=<psd from another writer> (a 40×30 canvas with a red 20×10
    // box at (8, 5), as made by psd-tools in the dev notes)
    let Ok(path) = std::env::var("WETDOC_PSD_EXT") else { return };
    let psd = wet_doc::psd::read_psd(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!((psd.w, psd.h), (40, 30));
    let l = &psd.layers[0];
    assert_eq!((l.left, l.top, l.w, l.h), (8, 5, 20, 10));
    assert_eq!(&l.rgba[..4], &[200, 30, 30, 255]);
}

#[test]
fn reference_images_round_trip() {
    let mut doc = Document::new(16, 16);
    doc.references.push(wet_doc::ReferenceImage { name: "ref".into(), w: 2, h: 1, rgba: vec![1, 2, 3, 4, 5, 6, 7, 8], visible: false });
    let mut bytes = Vec::new();
    doc.save(&mut bytes).unwrap();
    let back = Document::load(&bytes[..]).unwrap();
    assert_eq!(back.references[0].rgba, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(!back.references[0].visible);
}

#[test]
fn soft_proof_flags_out_of_gamut_paint() {
    use wet_color::RgbSpace;
    let mut doc = Document::new(64, 64);
    doc.assign_profile(Some(RgbSpace::DisplayP3));
    blob(&mut doc, 32.0, [10.0, 250.0, 10.0]); // saturated P3 green
    let (img, warn) = doc.soft_proof(&RgbSpace::Srgb.into());
    assert_eq!(img.len(), 64 * 64 * 4);
    assert!(warn[30 * 64 + 32], "saturated green should be out of sRGB");
    assert!(!warn[5 * 64 + 5], "paper is in gamut");
}
