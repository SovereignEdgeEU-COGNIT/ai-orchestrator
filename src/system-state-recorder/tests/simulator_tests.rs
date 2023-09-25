// use crate::core::Simulator;
use simulator::core::Simulator;

#[test]
fn test_new() {
    let simulator = Simulator::new();
    assert!(
        simulator.locations().is_empty(),
        "New simulator should have no locations."
    );
}

#[test]
fn test_add_location() {
    let mut simulator = Simulator::new();
    simulator.add_location("Milky Way".to_string());

    assert_eq!(simulator.locations().len(), 1);
    assert_eq!(simulator.locations()[0], "Milky Way");
}

#[test]
fn test_locations() {
    let mut simulator = Simulator::new();
    simulator.add_location("Location1".to_string());
    simulator.add_location("Location2".to_string());

    let locs = simulator.locations();
    assert_eq!(locs.len(), 2);
    assert!(locs.contains(&"Location1".to_string()));
    assert!(locs.contains(&"Location2".to_string()));
}
