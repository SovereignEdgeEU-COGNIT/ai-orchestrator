#!/bin/bash

curl -X POST "http://localhost:8000/addhost/?hostid=1&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=2&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=3&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=4&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=5&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=6&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=7&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=8&mem=8073741824&cpu=16000"
curl -X POST "http://localhost:8000/addhost/?hostid=9&mem=8073741824&cpu=16000"

curl -X PUT "http://localhost:8000/set?hostid=2&renewable=true"
curl -X PUT "http://localhost:8000/set?hostid=3&renewable=true"
