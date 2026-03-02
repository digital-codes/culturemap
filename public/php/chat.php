<?php

require_once __DIR__ . '/kekabo.php';

/*
https://docs.mistral.ai/agents/agents

start:
curl --location "https://api.mistral.ai/v1/conversations" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "inputs": [
       {
         "role": "user",
         "content": "Who is Albert Einstein?",
         "object": "entry",
         "type": "message.input"
       }
     ],
     "stream": false,
     "agent_id": "<agent_id>"
  }'

continue: 
curl --location "https://api.mistral.ai/v1/conversations/<conv_id>" \
     --header 'Content-Type: application/json' \
     --header 'Accept: application/json' \
     --header "Authorization: Bearer $MISTRAL_API_KEY" \
     --data '{
     "inputs": "Translate to French.",
     "stream": false,
     "store": true,
     "handoff_execution": "server"
  }'



*/


// Load configuration from config.ini if it exists
if (file_exists("/var/www/files/culturemap/config.ini")) {
    $configFile = "/var/www/files/culturemap/config.ini";
} else {
    $configFile = __DIR__ . '/config.ini';
}

if (file_exists($configFile)) {
    $config = parse_ini_file($configFile, true);
    print_r($config);
    $apiKey = $config['agent']['api_key'] ?? 'YOUR_MISTRAL_API_KEY';
    $agentId = $config['agent']['agent_id'] ?? 'YOUR_AGENT_ID';
} else {
    $apiKey = 'YOUR_MISTRAL_API_KEY';
    $agentId = 'YOUR_AGENT_ID';
}

$endpoint = 'https://api.mistral.ai/v1/conversations';

echo "Using endpoint $endpoint\n";

// Example chat payload

$currentDate = date('d.m.Y');

$lang = "Deutsch";

$query = "Heute ist der $currentDate. ";
$query .= " Die Sprache des Nutzers ist $lang. ";
//$query .= " Was liegt an am nächsten Wochenende?";
$query .= " Am Wochenende soll die Eröffnung der Le Cage Ausstellung sein. wann genau?";
//$query .= " Was passiert im Iran?";
//$query .= " What's up the next two days at the zkm?";
// $query .= " What are Trump's next plans?";


$payload = [
    'inputs' => [
        [
            'role' => 'user', 
            'content' => $query,
            'object' => 'entry',
            'type' => 'message.input'
        ]
    ],
    'stream' => false,
    "agent_id" => $agentId
];

$ch = curl_init($endpoint);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'Authorization: Bearer ' . $apiKey
]);

$response = curl_exec($ch);

print_r($response);
echo "\n--------------------\n";

/*
Using endpoint https://api.mistral.ai/v1/conversations
{"object":"conversation.response",
"conversation_id":"conv_019caeab2a14717f98fa1d9f6d3d1c3e",
"outputs":[{"object":"entry","type":"message.output",
"created_at":"2026-03-02T13:09:39.230436Z",
"completed_at":"2026-03-02T13:09:39.498332Z",
"agent_id":"ag_019c72abc7447066a80fb494889902a3",
"model":"mistral-medium-latest","id":"msg_019caeab2b1e7554afbc12f4fc750127",
"role":"assistant",
"content":"Dazu kann ich nichts sagen. Wie kann ich dir bei kulturellen Angeboten oder Initiativen in Karlsruhe helfen?"}],"usage":{"prompt_tokens":429,"completion_tokens":24,"total_tokens":453}}k

*/

$responseData = json_decode($response, true);

// check if the respone is a tool call for get_upcoming_events and if so, extract the events
// Check if the response contains a function call for get_upcoming_events
if (isset($responseData['outputs'])) {
    foreach ($responseData['outputs'] as $output) {
        if ($output['type'] === 'function.call' && $output['name'] === 'get_upcoming_events') {
            echo "Function call detected: get_upcoming_events\n";
            echo "Tool call ID: " . $output['tool_call_id'] . "\n";
            
            $arguments = json_decode($output['arguments'], true);
            echo "Start date: " . $arguments['start_date'] . "\n";
            echo "End date: " . $arguments['end_date'] . "\n\n";
            
            $timezone = new DateTimeZone('Europe/Berlin');
            $startDateTime = DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['start_date'], $timezone);
            $endDateTime = DateTime::createFromFormat('Y-m-d\TH:i:s', $arguments['end_date'], $timezone);

            if ($startDateTime !== false && $endDateTime !== false) {
                $startTimestamp = $startDateTime->getTimestamp();
                $endTimestamp = $endDateTime->getTimestamp();

                echo "Start timestamp: " . $startTimestamp . "\n";
                echo "End timestamp: " . $endTimestamp . "\n\n";

                $events = getEvents($startTimestamp, $endTimestamp);
                echo "Fetched " . count($events) . " events from getEvents function.\n";
                print_r($events);

                $response = send_event_list_to_agent($responseData['conversation_id'], $apiKey, $output['tool_call_id'], $events);  
                print_r($response);

            } else {
                echo "Error: Invalid date format in arguments\n";
            }

            // Here you would call your actual function to get events
            // $events = get_upcoming_events($arguments['start_date'], $arguments['end_date']);
        }
    }
}


// Initialize variables
$concatenatedText = '';
$webSearchResults = [];

// Loop through outputs
foreach ($responseData['outputs'] as $output) {
    if ($output['type'] === 'message.output' && isset($output['content'])) {
        if (is_string($output['content'])) {
            $concatenatedText .= $output['content'];
            continue;
        }
        foreach ($output['content'] as $content) {
            if ($content['type'] === 'text') {
                $concatenatedText .= $content['text'];
            } elseif ($content['type'] === 'tool_reference' && $content['tool'] === 'web_search') {
                $webSearchResults[] = [
                    'title' => $content['title'],
                    'url' => $content['url']
                ];
            }
        }
    }
}

// Output the concatenated text
echo "Concatenated Text:\n";
echo $concatenatedText . "\n\n";

// Output the web search results
echo "Web Search Results:\n";
print_r($webSearchResults);

function send_event_list_to_agent($conv_id, $mistral_api_key, $tool_call_id, $event_list) {
    $api_url = "https://api.mistral.ai/v1/conversations/" . $conv_id;

    $headers = [
        'Content-Type: application/json',
        'Accept: application/json',
        'Authorization: Bearer ' . $mistral_api_key
    ];

    $payload = [
        'inputs' => [
            [
                'tool_call_id' => $tool_call_id,
                'result' => json_encode($event_list),
                'object' => 'entry',
                'type' => 'function.result'
            ]
        ],
        'stream' => false,
        'store' => true,
        'handoff_execution' => 'server'
    ];

    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $api_url);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($ch);

    if (curl_errno($ch)) {
        return ['error' => 'Curl error: ' . curl_error($ch)];
    }

    return json_decode($response, true);
}


?>
