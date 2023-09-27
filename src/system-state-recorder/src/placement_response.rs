#![allow(non_snake_case)]
use serde::Deserialize;
use serde_json::Result;

#[derive(Deserialize, Debug)]
pub struct VM {
    pub ID: i32,
    pub HOST_ID: i32,
}

#[derive(Deserialize, Debug)]
pub struct Data {
    pub VMS: Vec<VM>,
}

pub fn parse_placement_response_json(json_str: &str) -> Result<Data> {
    serde_json::from_str(json_str)
}
