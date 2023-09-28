#!/bin/bash

curl -X POST "http://localhost:8000/placevm?vmid=1&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=2&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=3&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=4&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=5&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=6&mem=1073741824&cpu=2000"
curl -X POST "http://localhost:8000/placevm?vmid=7&mem=1073741824&cpu=2000"
