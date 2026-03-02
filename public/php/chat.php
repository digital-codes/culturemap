<?php
declare(strict_types=1);

/**
 * Simple HTTP handler for:
 *   POST /chat
 * JSON payload:
 *   - query (string, required)
 *   - session (int|string, optional)
 *   - conversation_id (string, optional)
 *
 * Response JSON:
 *   - error (int) 0 = ok
 *   - text (string)
 *   - session (int|string) new or existing session
 *   - conversation_id (string|null)
 */

header('Content-Type: application/json; charset=utf-8');

// --- Load your agent code ---
require_once __DIR__ . '/agent.php'; // adjust path if needed

// --- Log query to JSON file ---
$logFile = __DIR__ . '/queries.json';
$existingLogs = [];
if (file_exists($logFile)) {
    $logContent = file_get_contents($logFile);
    if ($logContent !== false) {
        $decoded = json_decode($logContent, true);
        if (is_array($decoded)) {
            $existingLogs = $decoded;
        }
    }
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
$query = isset($data['query']) ? trim((string)$data['query']) : '';
if ($query === '') {
    http_response_code(400);
    echo json_encode(['error' => 10, 'text' => 'Missing required field: query', 'session' => 0, 'conversation_id' => $data['conversation_id'] ?? null], JSON_UNESCAPED_UNICODE);
    exit;
}

// get language
$lang = isset($data['lang']) ? trim((string)$data['lang']) : 'DE';

if ($lang === 'DE' || $lang === 'de' || $lang === 'de-DE') {
    $lang = 'Deutsch';
} else if ($lang === 'EN' || $lang === 'en' || $lang === 'en-US') {
    $lang = 'Englisch';
} else {
    $lang = 'Deutsch'; // default to German if unrecognized
}

$sessionProvided = $data['session'] ?? null;

// Treat missing/0/"0"/"" as "new session"
$isNewSession = false;
if ($sessionProvided === null) {
    $isNewSession = true;
} elseif (is_int($sessionProvided) && $sessionProvided === 0) {
    $isNewSession = true;
} elseif (is_string($sessionProvided) && (trim($sessionProvided) === '' || trim($sessionProvided) === '0')) {
    $isNewSession = true;
}

if ($isNewSession) {
    // For follow-up messages, you might want to include session info in the query or handle it in your agent code
    if ($lang === 'Deutsch') {
        $query = "Heute ist der " . date('d.m.Y') . ". Die Sprache des Benutzers ist " . $lang . ". Die Frage ist: " . $query;
    } else {
        $query = "Today is " . date('F j, Y') . ". The user's language is " . $lang . ". The question is: " . $query;
    }
} 

$sessionOut = $sessionProvided;
$sessionForAgentHandler = 0;

// If new session, generate unique random numeric session id for the client response
if ($isNewSession) {
    // 31-bit positive int keeps it JSON-friendly across platforms
    $sessionOut = random_int(1, 2147483647);
    $sessionForAgentHandler = 0; // critical: session_id=0 starts a new chat
} else {
    // Follow-up: keep the client session value as-is, but pass a non-0 session_id into chat.php
    // chat.php casts session_id to int; ensure it's non-0 even if the client provided text.
    $sessionForAgentHandler = 1;
}

$conversationId = isset($data['conversation_id']) ? (string)$data['conversation_id'] : null;
if ($conversationId !== null && trim($conversationId) === '') {
    $conversationId = null;
}

// log the query with timestamp, session, conversation_id, and language
$logEntry = [
    'timestamp' => date('c'),
    'session' => $sessionOut,
    'conversation_id' => $conversationId,
    'lang' => $lang,
    'query' => $query,
];
$existingLogs[] = $logEntry;
file_put_contents($logFile, json_encode($existingLogs, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE), LOCK_EX);

// --- Call agent handler (session_id controls reset behavior) ---
try {
    $agentResult = handleHttpChatRequest([
        'query' => $query,
        'session_id' => $sessionForAgentHandler,
        'conversation_id' => $conversationId,
        // keep any other fields you support (language, etc.) out unless needed
    ]);

    $error = isset($agentResult['error']) ? (int)$agentResult['error'] : 99;

    // Always return a sensible text
    $text = '';
    if (isset($agentResult['text']) && is_string($agentResult['text']) && trim($agentResult['text']) !== '') {
        $text = $agentResult['text'];
    } else {
        $text = ($error === 0) ? 'OK' : 'An error occurred';
    }

    // Always transmit conversation_id back to client
    $conversationIdOut = $agentResult['conversation_id'] ?? $conversationId;

    http_response_code(200);
    echo json_encode([
        'error' => $error,
        'text' => $text,
        'session' => $sessionOut,
        'conversation_id' => $conversationIdOut,
    ], JSON_UNESCAPED_UNICODE);
} catch (Throwable $e) {
    http_response_code(500);
    echo json_encode([
        'error' => 99,
        'text' => 'Internal error: ' . $e->getMessage(),
        'session' => ($isNewSession ? $sessionOut : $sessionProvided),
        'conversation_id' => $conversationId,
    ], JSON_UNESCAPED_UNICODE);
}