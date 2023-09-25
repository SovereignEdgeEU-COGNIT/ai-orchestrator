mod request_handler;
#[macro_use]
extern crate rocket;

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![request_handler::index])
}
