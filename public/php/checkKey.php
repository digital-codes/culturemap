<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

// Load configuration
$configFile = file_exists("/var/www/files/culturemap/config.ini")
    ? "/var/www/files/culturemap/config.ini"
    : __DIR__ . '/config.ini';

if (file_exists($configFile)) {
    $config = parse_ini_file($configFile, true);
    $passkey = $config['edit']['passkey'] ?? 'YOUR_PASSKEY';
} else {
    http_response_code(500);
    die(json_encode(['error' => 'Configuration file not found.'])); 
}

// --- Basic routing (works for PHP built-in server or simple setups) ---
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH) ?? '/';


if ($method !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 405, 'text' => 'Method not allowed', 'session' => 0, 'conversation_id' => null], JSON_UNESCAPED_UNICODE);
    exit;
}

// --- Read + decode JSON body ---
$raw = file_get_contents('php://input');
if ($raw === false || trim($raw) === '') {
    http_response_code(400);
    echo json_encode(['error' => 400, 'text' => 'Missing JSON body', 'session' => 0, 'conversation_id' => null], JSON_UNESCAPED_UNICODE);
    exit;
}

$data = json_decode($raw, true);
if (!is_array($data)) {
    http_response_code(400);
    echo json_encode(['error' => 400, 'text' => 'Invalid JSON', 'session' => 0, 'conversation_id' => null], JSON_UNESCAPED_UNICODE);
    exit;
}

// --- Validate params ---
$pk = isset($data['passkey']) ? trim((string)$data['passkey']) : '';
if ($pk !== $passkey) {
    http_response_code(401);
    echo json_encode(['error' => 10, 'text' => 'Invalid passkey', 'session' => 0, 'conversation_id' => $data['conversation_id'] ?? null], JSON_UNESCAPED_UNICODE);
    exit;
}
