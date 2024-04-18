package database

import (
	"io"
	"log"
)

func PrepareTests() (*Database, error) {
	return PrepareTestsWithPrefix("TEST_")
}

func PrepareTestsWithPrefix(prefix string) (*Database, error) {
	log.SetOutput(io.Discard)

	dbHost := "localhost"
	dbPort := 5432
	dbUser := "postgres"
	dbPassword := "rFcLGNkgsNtksg6Pgtn9CumL4xXBQ7"
	dbName := "postgres"
	dbPrefix := prefix

	db := CreateDatabase(dbHost, dbPort, dbUser, dbPassword, dbName, dbPrefix)

	err := db.Connect()
	if err != nil {
		return nil, err
	}

	db.Drop()
	err = db.Initialize()

	return db, err
}
