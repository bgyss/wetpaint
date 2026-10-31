//! Desktop transports for the [`agent`](crate::agent) ops: JSON lines on stdio, a local
//! TCP listener whose requests the window's event loop answers, and a client for it.

use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

use serde_json::{Value, json};

use crate::{agent, app::App};

/// The address `--listen` uses without one: this machine only.
pub const DEFAULT_ADDR: &str = "127.0.0.1:7878";

/// A request line and where its reply line goes.
pub type Request = (String, mpsc::Sender<String>);

/// Accept connections on `addr`; each request line arrives on the receiver, to be answered
/// on the app's thread with [`agent::handle_line`].
pub fn listen(addr: &str) -> io::Result<(SocketAddr, mpsc::Receiver<Request>)> {
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?;
    let (tx, rx) = mpsc::channel::<Request>();
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let tx = tx.clone();
            thread::spawn(move || {
                let _ = connection(stream, tx);
            });
        }
    });
    Ok((local, rx))
}

fn connection(stream: TcpStream, tx: mpsc::Sender<Request>) -> io::Result<()> {
    let mut out = stream.try_clone()?;
    for line in BufReader::new(stream).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        if tx.send((line, reply_tx)).is_err() {
            break;
        }
        let Ok(reply) = reply_rx.recv() else { break };
        writeln!(out, "{reply}")?;
    }
    Ok(())
}

/// Answer request lines from stdin on stdout until stdin ends or `quit`.
pub fn stdio(app: &mut App) -> io::Result<()> {
    let mut out = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        writeln!(out, "{}", agent::handle_line(app, &line))?;
        out.flush()?;
        if app.quit {
            break;
        }
    }
    Ok(())
}

/// A client for a window listening on `addr`: sends a request, waits for its reply.
pub fn connect(addr: &str) -> io::Result<impl FnMut(&Value) -> Value + use<>> {
    let stream = TcpStream::connect(addr)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    Ok(move |req: &Value| {
        let mut line = String::new();
        let sent = writeln!(writer, "{req}").and_then(|_| reader.read_line(&mut line));
        match sent {
            Ok(n) if n > 0 => serde_json::from_str(&line).unwrap_or_else(|e| json!({ "ok": false, "error": format!("bad reply: {e}") })),
            Ok(_) => json!({ "ok": false, "error": "wetpaint closed the connection" }),
            Err(e) => json!({ "ok": false, "error": format!("lost the connection to wetpaint: {e}") }),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_over_tcp_are_answered_on_the_app_thread() {
        let (addr, rx) = listen("127.0.0.1:0").unwrap();
        let mut app = agent::headless(64, 48);
        let client = thread::spawn(move || {
            let mut call = connect(&addr.to_string()).unwrap();
            let a = call(&json!({ "op": "command", "name": "new_layer" }));
            let b = call(&json!({ "op": "state" }));
            (a, b)
        });
        for _ in 0..2 {
            let (line, reply) = rx.recv().unwrap();
            reply.send(agent::handle_line(&mut app, &line)).unwrap();
        }
        let (a, b) = client.join().unwrap();
        assert_eq!(a["ok"], true);
        assert_eq!(b["layers"].as_array().unwrap().len(), 2);
    }
}
