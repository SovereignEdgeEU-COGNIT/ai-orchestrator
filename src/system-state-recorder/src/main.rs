mod request_handler;

#[macro_use]
extern crate rocket;

mod monitor;
mod prometheus;
mod simulator;

use crate::monitor::Monitor;
use crate::prometheus::PrometheusMonitor;
use crate::simulator::{Simulator, SimulatorHelper};
use rocket::fairing::{Fairing, Info, Kind};
use rocket::http::Header;
use rocket::{Request, Response};
use std::sync::Arc;

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
    let mut simulator = Simulator::new();
    simulator.add_host_with_vms("1".to_string(), vec!["1".to_string(), "2".to_string()]);
    simulator.add_host_with_vms("2".to_string(), vec!["3".to_string()]);
    let monitor: Arc<dyn Monitor + Send> = Arc::new(simulator);

    //let monitor: Arc<dyn Monitor + Send> = Arc::new(PrometheusMonitor);

    rocket::build().manage(monitor).attach(Cors).mount(
        "/",
        routes![
            request_handler::index,
            request_handler::set_renewable,
            request_handler::get_host_info
        ],
    )
}
