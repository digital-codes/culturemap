<?php
// api.php
require_once 'dbCreate.php';
require_once 'llamaCheckToken.php';

header("Content-Type: application/json");
$db = new Database();
$conn = $db->getConnection();

$method = $_SERVER['REQUEST_METHOD'];
$input = json_decode(file_get_contents('php://input'), true);

if (!isset($input['token']) && in_array($method, ['POST', 'PUT'])) {
    http_response_code(400);
    echo json_encode(['error' => 'Token is required']);
    exit;
}   

switch ($method) {
    case 'GET':
        if (isset($_GET['id'])) {
            getEntry($conn, $_GET['id']);
        } else if (isset($_GET['img'])) {
            getByImg($conn, $_GET['img']);  
        } else {
            getAllEntries($conn);
        }
        break;
    case 'POST':
        if (null === parseToken($input['token'] ?? '')) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            break;
        }
        addEntry($conn, $input);
        break;
    case 'PUT':
        if (null === parseToken($input['token'] ?? '')) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            break;
        }
        updateEntry($conn, $input);
        break;
    case 'DELETE':
        if (null === parseToken($_GET['token'] ?? '')) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            break;
        }
        deleteEntry($conn, $_GET['id']);
        break;
    default:
        http_response_code(405);
        echo json_encode(['error' => 'Method not allowed']);
        break;
}

function getAllEntries($conn) {
    try {
        $stmt = $conn->prepare("SELECT * FROM entries");
        $stmt->execute();
        $entries = $stmt->fetchAll(PDO::FETCH_ASSOC);
        echo json_encode($entries);
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to fetch entries']);
    }
}

function getEntry($conn, $id) {
    try {
        if (!is_numeric($id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid ID format']);
            return;
        }
        $stmt = $conn->prepare("SELECT * FROM entries WHERE id = :id");
        $stmt->bindParam(':id', $id, PDO::PARAM_INT);
        $stmt->execute();
        $entry = $stmt->fetch(PDO::FETCH_ASSOC);
        if ($entry) {
            echo json_encode($entry);
        } else {
            http_response_code(404);
            echo json_encode(['error' => 'Entry not found']);
        }
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to fetch entry']);
    }
}

function getByImg($conn, $img) {
    try {
        $stmt = $conn->prepare("SELECT * FROM entries WHERE img = :img");
        $stmt->bindParam(':img', $img, PDO::PARAM_STR);
        $stmt->execute();
        $entry = $stmt->fetch(PDO::FETCH_ASSOC);
        if ($entry) {
            echo json_encode($entry);
        } else {
            http_response_code(404);
            echo json_encode(['error' => 'Entry not found']);
        }
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to fetch entry']);
    }
}

function addEntry($conn, $input) {
    try {
        if (!validateInput($input)) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing required fields']);
            return;
        }
        $stmt = $conn->prepare("
            INSERT INTO entries
            (name, url, location, geo_lat, geo_lng, img, description, tags)
            VALUES (:name, :url, :location, :geo_lat, :geo_lng, :img, :description, :tags)
        ");
        $stmt->bindParam(':name', $input['name']);
        $stmt->bindParam(':url', $input['url']);
        $stmt->bindParam(':location', $input['location']);
        $stmt->bindParam(':geo_lat', $input['geo'][0]);
        $stmt->bindParam(':geo_lng', $input['geo'][1]);
        $stmt->bindParam(':img', $input['img']);
        $stmt->bindParam(':description', $input['description']);
        $stmt->bindParam(':tags', $input['tags']);
        $stmt->execute();
        echo json_encode(['id' => $conn->lastInsertId()]);
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to create entry']);
    }
}

function updateEntry($conn, $input) {
    try {
        if (!isset($input['id'])) {
            http_response_code(400);
            echo json_encode(['error' => 'ID is required']);
            return;
        }
        $stmt = $conn->prepare("
            UPDATE entries SET
            name = :name,
            url = :url,
            location = :location,
            geo_lat = :geo_lat,
            geo_lng = :geo_lng,
            img = :img,
            description = :description,
            tags = :tags
            WHERE id = :id
        ");
        $stmt->bindParam(':id', $input['id']);
        $stmt->bindParam(':name', $input['name']);
        $stmt->bindParam(':url', $input['url']);
        $stmt->bindParam(':location', $input['location']);
        $stmt->bindParam(':geo_lat', $input['geo'][0]);
        $stmt->bindParam(':geo_lng', $input['geo'][1]);
        $stmt->bindParam(':img', $input['img']);
        $stmt->bindParam(':description', $input['description']);
        $stmt->bindParam(':tags', $input['tags']);
        $stmt->execute();
        if ($stmt->rowCount() === 0) {
            http_response_code(404);
            echo json_encode(['error' => 'Entry not found']);
        } else {
            echo json_encode(['success' => true]);
        }
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to update entry']);
    }
}

function deleteEntry($conn, $id) {
    try {
        if (!is_numeric($id)) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid ID format']);
            return;
        }
        $stmt = $conn->prepare("DELETE FROM entries WHERE id = :id");
        $stmt->bindParam(':id', $id, PDO::PARAM_INT);
        $stmt->execute();
        if ($stmt->rowCount() === 0) {
            http_response_code(404);
            echo json_encode(['error' => 'Entry not found']);
        } else {
            echo json_encode(['success' => true]);
        }
    } catch (PDOException $e) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to delete entry']);
    }
}

function validateInput($input) {
    return isset($input['name'], $input['location'], 
                 $input['img'], $input['description']);
}
