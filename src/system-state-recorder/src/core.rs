pub struct Simulator {
    locations: Vec<String>,
}

impl Simulator {
    pub fn new() -> Self {
        Simulator {
            locations: Vec::new(),
        }
    }

    pub fn locations(&self) -> &Vec<String> {
        &self.locations
    }

    pub fn add_location(&mut self, loc: String) {
        self.locations.push(loc);
    }
}
