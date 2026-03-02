<?php

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
$query .= " Was liegt an am nächsten Wochenende?";
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



?>
