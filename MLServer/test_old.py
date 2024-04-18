import pytest
from flask_testing import TestCase
from server import create_app, CustomFlask

class TestFlaskApp(TestCase):
    def create_app(self):
        app = create_app(__name__)
        app.config['TESTING'] = True
        app.config['DEBUG'] = True
        return app
    
    def test_api_place_valid_request(self):
        """Integration test for valid placement requests."""
        valid_data = {'ID': 1, 'HOST_IDS': [7]}
        response = self.client.post('/api/place', json=valid_data)
        self.assert200(response)
        self.assertIn('HOST_ID', response.json)
        self.assertNotEmpty(response.json['HOST_ID'])
        self.assertEquals(response.json['HOST_ID'], 7)

    def test_api_place_invalid_request(self):
        """Integration test for invalid placement requests."""
        invalid_data = {}  # Missing required fields
        response = self.client.post('/api/place', json=invalid_data)
        self.assert400(response)
        
    def test_api_place_invalid_hosts(self):
        """Integration test for invalid placement requests."""
        invalid_data = {'ID': 1, 'HOST_IDS': [700, 800]}  # Missing required fields
        response = self.client.post('/api/place', json=invalid_data)
        self.assertIn('HOST_ID', response.json)
        self.assertNotEmpty(response.json['HOST_ID'])
        self.assertEquals(response.json['HOST_ID'], -1)

    def test_api_scheduler_set(self):
        """Integration test for setting a scheduler model."""
        data = {'model_name': 'InteferenceAwareScheduler'}
        response = self.client.post('/api/scheduler', json=data)
        self.assert200(response)
        self.assertEqual(response.json['status'], 'success')

    def test_api_classifier_set(self):
        """Integration test for setting a classifier model."""
        data = {'model_name': 'DLClassifier'}
        response = self.client.post('/api/classifier', json=data)
        self.assert200(response)
        self.assertEqual(response.json['status'], 'success')

    def test_api_iaEnergyScalar_set(self):
        """Integration test for setting IA energy scalar."""
        data = {'energy_cost_scalar': 1.5}
        response = self.client.post('/api/iaEnergyScalar', json=data)
        self.assert200(response)
        self.assertEqual(response.json['status'], 'success')

if __name__ == '__main__':
    pytest.main()
