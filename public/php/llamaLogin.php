<?php

declare(strict_types=1);
/* access control handled in site config. don't duplicate !!!
header("Access-Control-Allow-Headers: *");
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
 */

use Lcobucci\JWT\Encoding\ChainedFormatter;
use Lcobucci\JWT\Encoding\JoseEncoder;
use Lcobucci\JWT\Signer\Key\InMemory;
//use Lcobucci\JWT\Signer\Hmac\Sha256;
use Lcobucci\JWT\Signer\Rsa\Sha256;
use Lcobucci\JWT\Token\Builder;

// for validation
use Lcobucci\JWT\Token\Parser;
use Lcobucci\JWT\Validation\Validator;
use Lcobucci\JWT\Validation\Constraint\SignedWith;
use Lcobucci\JWT\Validation\Constraint\IdentifiedBy;
use Lcobucci\JWT\Validation\Constraint\LooseValidAt;
use Lcobucci\JWT\Encoding\CannotDecodeContent;
use Lcobucci\JWT\Token\InvalidTokenStructure;
use Lcobucci\JWT\Token\UnsupportedHeaderFound;
use Lcobucci\JWT\UnencryptedToken;
use Lcobucci\Clock\SystemClock;
//

require 'vendor/autoload.php';

$basedirFile = __DIR__ . '/basedir.txt';
$basedir = '/var/www/files/';

if (is_readable($basedirFile)) {
  $contents = @file_get_contents($basedirFile);
  if ($contents !== false) {
    $contents = trim($contents);
    // remove surrounding quotes if present
    $contents = trim($contents, "\"' \t\n\r\0\x0B");
    if ($contents !== '') {
      $basedir = $contents;
    }
  }
}
// ensure trailing slash
$basedir = rtrim($basedir, "/\\") . '/';

// upgrade to public key
// cors proxy login
// Define the CSV file path
// $csv_file = 'users.csv';
// on localhost we can use simple credentials
$csv_file = "{$basedir}users.csv";
$public_file = "{$basedir}public.pem";
$private_file = "{$basedir}private.pem";
$access_file = "{$basedir}access.ini";
// overwrite files on localhost
/*
if (($_SERVER === null) || ($_SERVER['SERVER_NAME'] === 'localhost') || ($_SERVER['REMOTE_ADDR'] === '127.0.0.1')) {
  $csv_file = "./users.csv";    
  $public_file = "./public.pem";
  $private_file = "./private.pem";
  $access_file = "./access.ini";
}
*/

function getJtiClaim() {
  global $access_file;
  $access = @parse_ini_file($access_file, true, INI_SCANNER_TYPED);
  if ($access === false) {
      http_response_code(500);
      echo json_encode(["error" => "Internal server error (no access file)"]);
      exit;
    }
  if (empty($access['JTI'])) {
      http_response_code(500);
      echo json_encode(["error" => "Internal server error (no jti)"]);
      exit;
  }
  // set claim
  return trim((string)$access['JTI']);
}


function sanitizeUsername($username) {
  $username = filter_var($username, FILTER_SANITIZE_STRING);
  $username = trim($username);
  return $username;
}

function sanitizePassword($password) {
  $password = filter_var($password, FILTER_SANITIZE_STRING);
  $password = trim($password);
  return $password;
}

function extractUsernameAndPassword() {
    $username = '';
    $password = '';
  
    // Check if the request method is POST
    if ($_SERVER['REQUEST_METHOD'] === 'POST') {
      // Get the raw POST data
      $postData = file_get_contents('php://input');
      $params = json_decode($postData, true);

      // Extract the username and password from the parameters
      if (isset($params['username']) && isset($params['password'])) {
        $username = $params['username'];
        $password = $params['password'];
      }
    }
  
    return array('username' => $username, 'password' => $password);
  }

function parseToken($token) {
  checkToken($token);
  $parser = new Parser(new JoseEncoder());

  try {
      $tok = $parser->parse($token);
  } catch (CannotDecodeContent | InvalidTokenStructure | UnsupportedHeaderFound $e) {
      http_response_code(401);
      echo json_encode(array("error" => "Invalid username or password"));
      die();
  }
  assert($tok instanceof UnencryptedToken);
  // check jti claim
  $jti = getJtiClaim();
  if ($tok->claims()->get('jti') !== $jti) {
      http_response_code(401);
      echo json_encode(array("error" => "Invalid username or password (JTI)"));
      die();
  }

  $user = $tok->claims()->get('uid'); 
  //echo("User: " . $user . PHP_EOL);
  return $user;
  
}

function checkToken($token,$publicKey) {
  $parser = new Parser(new JoseEncoder());
  $tok = $parser->parse($token);

  // Create a key object from the public key
  $check = InMemory::plainText($publicKey);

  // Verify the signature using the SHA-256 algorithm and the public key
  $validator = new Validator();
  $algorithm    = new Sha256();

  if (! $validator->validate($tok,  new SignedWith($algorithm,$check))) {
      http_response_code(401);
      echo json_encode(array("error" => "Invalid username or password"));
      die();
  }

  $clock = new SystemClock(new DateTimeZone(date_default_timezone_get()));
  if (! $validator->validate($tok, new LooseValidAt($clock))) {
      http_response_code(401);
      echo json_encode(array("error" => "Invalid username or password"));
      die();
  }
}

// https://lcobucci-jwt.readthedocs.io/en/4.3.0/issuing-tokens/  
function makeToken($username) {
  global $public_file, $private_file;

  //echo("Files: " . $public_file . PHP_EOL);
  // read public and private key
  $privateKey = file_get_contents($private_file);

  //echo( "Public:" . $publicKey . PHP_EOL);
  //echo( "Private:" . $privateKey . PHP_EOL);
  
  $tokenBuilder = (new Builder(new JoseEncoder(), ChainedFormatter::default()));
  $verifyKey = InMemory::plainText($privateKey); // 
  $algorithm    = new Sha256();
  
  $now   = new DateTimeImmutable();
  $token = $tokenBuilder
      // Configures the issuer (iss claim)
      ->issuedBy('https://culture-map.space')
      // Configures the audience (aud claim)
      ->permittedFor('https://culture-map.space')
      // Configures the id (jti claim)
      ->identifiedBy(getJtiClaim())
      // Configures the time that the token was issue (iat claim)
      ->issuedAt($now)
      // Configures the time that the token can be used (nbf claim)
      ->canOnlyBeUsedAfter($now->modify('-1 hour'))  // client should use UTC but not guaranteed
      // Configures the expiration time of the token (exp claim)
      ->expiresAt($now->modify('+1 hour'))
      // Configures a new claim, called "uid"
      ->withClaim('uid', $username)
      // Configures a new header, called "foo"
      //->withHeader('foo', 'bar')
      // Builds a new token
      ->getToken($algorithm, $verifyKey);

      return $token->toString();
}

function login() {
    global $csv_file, $public_file, $private_file, $token_exp_time, $jti_claim, $access_file;

    // Extract the username and password from the request
    $params = extractUsernameAndPassword();
    $username = $params['username'];
    $password = $params['password'];
    
    // Check if the username and password are provided
    if (empty($username) || empty($password)) {
        http_response_code(400);
        echo json_encode(array("error" => "Missing username or password"));
        return;
    }
    
    // Sanitize the inputs
    $username = sanitizeUsername($username);
    $password = sanitizePassword($password);
    
    // Check if the CSV file exists
    if (!file_exists($csv_file)) {
        http_response_code(500);
        echo json_encode(array("error" => "Internal server error (no csv)"));
        return;
    }
    
    // Open the CSV file for reading
    if (($handle = fopen($csv_file, "r")) !== FALSE) {
        // Loop through the CSV rows
        while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
        // Check if the username and password match
        // password must behashed like so:
        // $hashed_pwd = ( password_hash($the_password, PASSWORD_DEFAULT ));
        // and verified like so
        // password_verify(user_input,$hashed_pwd);
        // true if OK
        if ($data[0] === $username && password_verify($password, $data[1])) {
        //  if ($data[0] === $username && ($password === $data[1])) {
            $token = makeToken($username); //JWT::encode($payload, $secret_key);
            $publicKey = file_get_contents($public_file);
            checkToken($token,$publicKey); // throws on error here 
            echo json_encode(array("token" => $token,"key" => $publicKey));
            return;
        }
        }
        fclose($handle);
    }
    
    // If we reach this point, the username or password is invalid
    http_response_code(401);
    echo json_encode(array("error" => "Invalid username or password (final)"));
    }


// ----------------------------------------------------
// 2️⃣  Handle preflight (OPTIONS)
// ----------------------------------------------------
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    // The browser is checking what’s allowed; we don’t send a body
    http_response_code(204); // No Content
    exit;
}

// ----------------------------------------------------
// 3️⃣  Handle login (POST)
// ----------------------------------------------------
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    login();
	exit;
}

// ----------------------------------------------------
// 4️⃣  For any other HTTP verb
// ----------------------------------------------------
http_response_code(405); // Method Not Allowed
echo json_encode(['error' => 'Only POST supported']);


?>



