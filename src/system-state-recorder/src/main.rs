mod request_handler;

#[macro_use]
extern crate rocket;

mod monitor;
mod prometheus;
mod simulator;

use crate::monitor::Monitor;
use crate::prometheus::PrometheusMonitor;
use crate::simulator::{Host, Simulator, SimulatorFactory};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::Header;
use rocket::{Request, Response};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "staterec", about = "system state recorder")]
struct Opt {
    #[structopt(short, long)]
    sim: bool,
    aiorchestrator: Option<String>,
}

pub struct Cors;

#[rocket::async_trait]
impl Fairing for Cors {
    fn info(&self) -> Info {
        Info {
            name: "Cross-Origin-Resource-Sharing Fairing",
            kind: Kind::Response,
        }
    }

    async fn on_response<'r>(&self, _request: &'r Request<'_>, response: &mut Response<'r>) {
        response.set_header(Header::new("Access-Control-Allow-Origin", "*"));
        response.set_header(Header::new(
            "Access-Control-Allow-Methods",
            "POST, PATCH, PUT, DELETE, HEAD, OPTIONS, GET",
        ));
        response.set_header(Header::new("Access-Control-Allow-Headers", "*"));
        response.set_header(Header::new("Access-Control-Allow-Credentials", "true"));
    }
}

#[launch]
fn rocket() -> _ {
    let opt = Opt::from_args();

    if opt.sim {
        if let Some(url) = opt.aiorchestrator {
            println!("aiorchestrator url={}", url);
        }

        let hosts: HashMap<String, Host> = HashMap::new();
        let shared_hosts = Arc::new(Mutex::new(hosts));
        let simulator: Simulator = SimulatorFactory::new(Arc::clone(&shared_hosts));
        let monitor: Arc<dyn Monitor + Send> = Arc::new(simulator);

        rocket::build()
            .manage(monitor)
            .manage(shared_hosts)
            .attach(Cors)
            .mount(
                "/",
                routes![
                    request_handler::index,
                    request_handler::set_renewable,
                    request_handler::get_host_info,
                    request_handler::add_host,
                    request_handler::add_vm,
                ],
            )
    } else {
        let monitor: Arc<dyn Monitor + Send> = Arc::new(PrometheusMonitor);
        rocket::build().manage(monitor).attach(Cors).mount(
            "/",
            routes![
                request_handler::index,
                request_handler::set_renewable,
                request_handler::get_host_info
            ],
        )
    }
}
