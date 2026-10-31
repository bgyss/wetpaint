//! Preset response curves (`<param>_curve` objects).
//!
//! `{"fromParams": bool, "function": 2, "points": [{"x","y"}, ...], ...}`. With
//! `fromParams` the curve follows the plain parameter (e.g. `pressure_size_influence`),
//! otherwise its points apply. Curves use `function` 2 (BEZIER), evaluated
//! as eased (smoothstep) segments between consecutive points, holding the end values
//! flat outside the point range.

use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub points: Vec<(f32, f32)>,
    pub output: (f32, f32),
}

impl Curve {
    /// Parse a curve that overrides its parameter (`fromParams: false`); `None` otherwise.
    pub fn from_json(v: &Value) -> Option<Self> {
        if v.get("fromParams").and_then(Value::as_bool).unwrap_or(true) {
            return None;
        }
        let mut points: Vec<(f32, f32)> = v
            .get("points")?
            .as_array()?
            .iter()
            .filter_map(|p| Some((p.get("x")?.as_f64()? as f32, p.get("y")?.as_f64()? as f32)))
            .collect();
        if points.is_empty() {
            return None;
        }
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let g = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
        Some(Self { points, output: (g("outputMin", 0.0), g("outputMax", 1.0)) })
    }

    pub fn eval(&self, x: f32) -> f32 {
        let p = &self.points;
        let y = if x <= p[0].0 {
            p[0].1
        } else if x >= p[p.len() - 1].0 {
            p[p.len() - 1].1
        } else {
            let i = p.windows(2).position(|w| x <= w[1].0).unwrap_or(0);
            let ((x0, y0), (x1, y1)) = (p[i], p[i + 1]);
            let t = ((x - x0) / (x1 - x0).max(1e-6)).clamp(0.0, 1.0);
            y0 + (y1 - y0) * t * t * (3.0 - 2.0 * t)
        };
        self.output.0 + (self.output.1 - self.output.0) * y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves() {
        let v: Value = serde_json::from_str(
            r#"{"fromParams": false, "function": 2, "outputMin": 0, "outputMax": 1,
                "points": [{"x": 0, "y": 0.5}, {"x": 0.5, "y": 1}]}"#,
        )
        .unwrap();
        let c = Curve::from_json(&v).unwrap();
        assert_eq!(c.eval(0.0), 0.5);
        assert_eq!(c.eval(0.25), 0.75);
        assert_eq!(c.eval(1.0), 1.0);
        let fp: Value = serde_json::from_str(r#"{"fromParams": true, "points": []}"#).unwrap();
        assert!(Curve::from_json(&fp).is_none());
    }
}
