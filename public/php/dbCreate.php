<?php
// database.php
class Database {
    private $db;

    public function __construct() {
        $this->db = new PDO('sqlite:culturemap.sqlite');
        $this->db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $this->createTable();
    }

    private function createTable() {
        $query = "
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            location TEXT NOT NULL,
            geo_lat REAL NOT NULL,
            geo_lng REAL NOT NULL,
            img TEXT NOT NULL UNIQUE,
            description TEXT
        )";
        $this->db->exec($query);
    }

    public function getConnection() {
        return $this->db;
    }
}
?>

