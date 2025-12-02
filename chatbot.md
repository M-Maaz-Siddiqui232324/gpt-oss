# FlowHCM Chatbot Implementation Documentation

## Overview
This document provides comprehensive documentation for the FlowHCM chatbot implementation, covering all components, flows, and authentication mechanisms.

## Architecture Overview

```
Frontend (Angular) 
    ↓ [HTTP Request with Token]
ChatbotController (HCMSAPI)
    ↓ [Validates Token & Client Access]
ChatbotBusiness (Business Layer)
    ↓ [Business Logic & Key Generation]
ChatbotRepository (Data Layer)
    ↓ [Database Operations via Stored Procedures]
MySQL Database (FlowHCM)
    ↓ [Sync API Key]
FastAPI (Python Chatbot Service)
    ↓ [Validates API Key]
PostgreSQL (Chatbot Middleware)
```

---

## Component Details

### 1. ChatbotController.cs (API Layer)

**Location:** `HCMSAPI/Controllers/ChatbotController.cs`

**Purpose:** Handles HTTP requests from frontend, validates user authentication, and forwards queries to FastAPI.

#### Key Methods:

##### 1.1 GetResponse() / GetResponseFunc()
**Endpoint:** `POST /api/Chatbot/GetResponse?usermsg={message}`

**Flow:**

1. **Extract Token from Headers**
   ```csharp
   var token = headers.GetValues("token").First();
   ```
   - Validates token exists
   - Returns 400 BadRequest if missing

2. **Retrieve User Data from Cache**
   ```csharp
   UserData userData = memCache.GetValue(memCache.CreateCacheKey(token));
   int clientId = userData.FkClientId;
   string username = userData.UserName;
   ```
   - Uses FlowHCM's memCache pattern
   - Extracts ClientID and Username

3. **Get Client API Key and CompanyPin**
   ```csharp
   var clientInfo = chatbotBusiness.GetClientInfo(clientId);
   string apiKey = clientInfo?.APIKey;
   string companyPin = clientInfo?.CompanyPin;
   ```
   - Calls business layer to fetch client credentials

4. **Validate Chatbot Access**
   ```csharp
   if (string.IsNullOrEmpty(apiKey)) {
       return "Chatbot access is not enabled for your organization";
   }
   ```
   - Checks if client has API key
   - Returns 403 Forbidden if no access

5. **Forward to FastAPI**
   ```csharp
   Dictionary<string, string> result = await ChatbotAPI(usermsg, clientId, companyPin, apiKey, username);
   ```
   - Sends query with authentication headers

6. **Process Response**
   - Converts Markdown to HTML using Markdig
   - Returns formatted response to frontend
   - Logs activity via tracer

**Error Handling:**
- Missing token → 400 BadRequest
- No API key → 403 Forbidden
- FastAPI error → 400 BadRequest with user-friendly message
- Exception → Generic error with support contact info

---

##### 1.2 ChatbotAPI()
**Purpose:** Forwards authenticated request to FastAPI chatbot service

**Parameters:**
- `msg` - User's chat message
- `clientId` - Client identifier
- `companyPin` - Client's company PIN
- `apiKey` - Client's API key
- `username` - Current user's username

**Flow:**

1. **Get FastAPI URL from Config**
   ```csharp
   var baseUrl = ConfigurationManager.AppSettings["FastAPIURL"] ?? "http://localhost:8000";
   ```

2. **Set Authentication Headers**
   ```csharp
   client.DefaultRequestHeaders.Add("X-Client-ID", clientId.ToString());
   client.DefaultRequestHeaders.Add("X-Company-Pin", companyPin);
   client.DefaultRequestHeaders.Add("X-API-Key", apiKey);
   client.DefaultRequestHeaders.Add("X-User-Name", username);
   ```

3. **Send POST Request**
   ```csharp
   var requestPayload = new { query = msg };
   HttpResponseMessage response = await client.PostAsync("/query", content);
   ```

4. **Parse Response**
   - Success: Extract `response` field from JSON
   - Failure: Return error message

**Returns:** Dictionary with `status_code`, `data` (on success), or `error` (on failure)

---

##### 1.3 ConvertMarkdownToHtml()
**Purpose:** Converts Markdown formatted responses to HTML for frontend display

**Implementation:**
```csharp
var pipeline = new MarkdownPipelineBuilder()
    .UseAdvancedExtensions() // Tables, task lists, etc.
    .Build();
return Markdown.ToHtml(markdown, pipeline);
```

**Features:**
- Supports tables, lists, code blocks
- Graceful fallback to original text on error

---

### 2. ChatbotBusiness.cs (Business Layer)

**Location:** `HCMS.Business/ChatbotBusiness.cs`

**Purpose:** Business logic for API key generation, client data retrieval, and PostgreSQL synchronization.

#### Key Methods:

##### 2.1 GetClientInfo()
**Purpose:** Retrieve client's API key and CompanyPin from database

**Flow:**

1. **Create Repository Instance**
   ```csharp
   using (ChatbotRepository repository = new ChatbotRepository())
   ```
   - Uses `using` statement for proper disposal

2. **Call Repository Method**
   ```csharp
   return repository.GetClientInfo(clientId);
   ```

3. **Error Handling**
   - Wraps exceptions with descriptive message
   - Throws exception up to controller

**Returns:** `ClientInfo` object with ClientID, CompanyPin, APIKey, IsActive

---

##### 2.2 GenerateAPIKeyForClient()
**Purpose:** Generate secure API key for a client and sync to PostgreSQL

**Parameters:**
- `clientId` - Client identifier
- `createdBy` - User ID who generated the key

**Flow:**

1. **Generate Secure Key**
   ```csharp
   string apiKey = GenerateSecureAPIKey(clientId);
   ```
   - Calls private method for key generation

2. **Save to MySQL**
   ```csharp
   string result = repository.GenerateAPIKey(clientId, apiKey, createdBy);
   ```
   - Stores in FlowHCM database via stored procedure

3. **Get Client Info**
   ```csharp
   var clientInfo = repository.GetClientInfo(clientId);
   string companyPin = clientInfo?.CompanyPin;
   ```
   - Retrieves CompanyPin for sync

4. **Sync to PostgreSQL (Async)**
   ```csharp
   Task.Run(async () => await SyncToPostgreSQL(clientId, companyPin, apiKey));
   ```
   - Fire-and-forget async operation
   - Doesn't block main thread

**Returns:** Generated API key string

---

##### 2.3 GenerateSecureAPIKey() [Private]
**Purpose:** Generate cryptographically secure API key using HMACSHA256

**Algorithm:**

1. **Create Message**
   ```csharp
   string message = $"{clientId}-{DateTime.UtcNow.Ticks}";
   ```
   - Combines ClientID with current timestamp
   - Ensures uniqueness

2. **Get Secret Key**
   ```csharp
   string secretKey = ConfigurationManager.AppSettings["APIKeySecret"] 
                      ?? "FlowHCM-Secret-Key-2025";
   ```
   - Reads from Web.config or uses default

3. **Generate HMAC Hash**
   ```csharp
   using (var hmac = new HMACSHA256(Encoding.UTF8.GetBytes(secretKey)))
   {
       byte[] hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(message));
   }
   ```
   - Uses HMACSHA256 for cryptographic security

4. **Format Key**
   ```csharp
   string key = Convert.ToBase64String(hashBytes)
       .Replace("+", "")
       .Replace("/", "")
       .Replace("=", "")
       .Substring(0, 40);
   
   return $"FLOW-{clientId}-{key}";
   ```
   - Removes special characters
   - Takes first 40 characters
   - Prefixes with `FLOW-{clientId}-`

**Example Output:** `FLOW-1-a3f8d9e2c1b4567890abcdef1234567890abcdef`

---

##### 2.4 SyncToPostgreSQL() [Private]
**Purpose:** Synchronize client API key to PostgreSQL database via FastAPI

**Flow:**

1. **Get FastAPI URL**
   ```csharp
   string fastApiUrl = ConfigurationManager.AppSettings["FastAPIURL"];
   ```
   - Throws exception if not configured

2. **Prepare Sync Data**
   ```csharp
   var syncData = new {
       client_id = clientId,
       company_pin = companyPin,
       api_key = apiKey,
       is_active = true
   };
   ```

3. **Send POST Request**
   ```csharp
   var response = await client.PostAsync($"{fastApiUrl}/admin/sync-client", content);
   ```
   - Endpoint: `/admin/sync-client`
   - Timeout: 30 seconds

4. **Log Result**
   - Success: Logs confirmation
   - Failure: Logs error with response content
   - Exception: Logs exception message

**Note:** This is a fire-and-forget operation. Failures don't block the main flow.

---

### 3. ChatbotRepository.cs (Data Layer)

**Location:** `HCMS.Data/Repository/ChatbotRepository.cs`

**Purpose:** Database operations for chatbot functionality using stored procedures.

#### Key Methods:

##### 3.1 GetClientInfo()
**Purpose:** Retrieve client API key and details from database

**Stored Procedure:** `usp_Chatbot_GetClientAPIKey`

**Flow:**

1. **Open Database Connection**
   ```csharp
   using (MySqlConnection conn = new MySqlConnection(constr))
   {
       conn.Open();
   }
   ```

2. **Execute Stored Procedure**
   ```csharp
   using (MySqlCommand cmd = new MySqlCommand("usp_Chatbot_GetClientAPIKey", conn))
   {
       cmd.CommandType = CommandType.StoredProcedure;
       cmd.Parameters.AddWithValue("@ClientID", clientId);
   }
   ```

3. **Read Results**
   ```csharp
   using (MySqlDataReader reader = cmd.ExecuteReader())
   {
       if (reader.Read())
       {
           clientInfo = new ClientInfo {
               ClientID = Convert.ToInt32(reader["ClientID"]),
               CompanyPin = reader["CompanyPin"]?.ToString(),
               APIKey = reader["APIKey"]?.ToString(),
               IsActive = Convert.ToBoolean(reader["IsActive"])
           };
       }
   }
   ```

4. **Return Result**
   - Returns `ClientInfo` object if found
   - Returns `null` if no matching record

**Returns:** `ClientInfo` or `null`

---

##### 3.2 GenerateAPIKey()
**Purpose:** Save generated API key to database

**Stored Procedure:** `usp_Chatbot_GenerateAPIKey`

**Parameters:**
- `@ClientID` - Client identifier
- `@APIKey` - Generated API key
- `@CreatedBy` - User who generated the key

**Flow:**

1. **Open Connection**
   ```csharp
   using (MySqlConnection conn = new MySqlConnection(constr))
   ```

2. **Execute Stored Procedure**
   ```csharp
   cmd.Parameters.AddWithValue("@ClientID", clientId);
   cmd.Parameters.AddWithValue("@APIKey", apiKey);
   cmd.Parameters.AddWithValue("@CreatedBy", createdBy);
   ```

3. **Read Result**
   ```csharp
   if (reader.Read())
   {
       result = reader["APIKey"]?.ToString();
   }
   ```

**Returns:** API key string if successful, `null` if failed

---

##### 3.3 IDisposable Implementation
**Purpose:** Proper resource cleanup

**Pattern:**

```csharp
~ChatbotRepository() { Dispose(false); }

public void Dispose()
{
    Dispose(true);
    GC.SuppressFinalize(this);
}

protected virtual void Dispose(bool disposing)
{
    if (!disposed)
    {
        if (disposing) { /* Cleanup managed resources */ }
        disposed = true;
    }
}
```

**Usage:** Ensures proper cleanup when used with `using` statements

---

#### Helper Classes

##### ClientInfo
**Purpose:** Data transfer object for client information

**Properties:**
```csharp
public class ClientInfo
{
    public int ClientID { get; set; }
    public string CompanyPin { get; set; }
    public string APIKey { get; set; }
    public bool IsActive { get; set; }
}
```

---

## Database Schema

### Stored Procedures

#### 1. usp_Chatbot_GetClientAPIKey

**Purpose:** Retrieve client API key with access validation

**Input Parameters:**
- `@ClientID` (INT) - Client identifier

**Output Columns:**
- `ClientID` (INT)
- `CompanyPin` (VARCHAR)
- `APIKey` (VARCHAR)
- `IsActive` (BOOLEAN)

**Logic:**

1. **Find Chatbot Module ID**
   ```sql
   SELECT pkApplicationModuleId INTO ChatbotModuleID 
   FROM applicationmodule 
   WHERE Name = 'Chatbot' AND IsActive = 1 
   LIMIT 1;
   ```

2. **Validate Client Access**
   ```sql
   SELECT c.pkClientId AS ClientID,
          c.CompanyPin,
          c.api_key AS APIKey,
          c.IsActive
   FROM client c
   INNER JOIN clientsmodule cm ON cm.fkClientID = c.pkClientId
   WHERE c.pkClientId = ClientID
     AND c.IsActive = 1
     AND c.api_key IS NOT NULL
     AND c.api_key != ''
     AND cm.fkApplicationModuleID = ChatbotModuleID
     AND cm.IsAccess = 1;
   ```

**Validation Rules:**
- Client must be active
- API key must exist and not be empty
- Chatbot module must be active
- Client must have access to Chatbot module

---

#### 2. usp_Chatbot_GenerateAPIKey

**Purpose:** Save or update client API key

**Input Parameters:**
- `@ClientID` (INT) - Client identifier
- `@APIKey` (VARCHAR) - Generated API key
- `@CreatedBy` (INT) - User who generated the key

**Output Columns:**
- `APIKey` (VARCHAR) - Saved API key

**Logic:**
```sql
UPDATE client 
SET api_key = @APIKey,
    ModifiedDate = NOW(),
    ModifiedBy = @CreatedBy
WHERE pkClientId = @ClientID;

SELECT api_key AS APIKey 
FROM client 
WHERE pkClientId = @ClientID;
```

---

### Database Tables

#### client
**Relevant Columns:**
- `pkClientId` (INT, PK) - Client identifier
- `CompanyPin` (VARCHAR) - Company PIN for authentication
- `api_key` (VARCHAR) - Chatbot API key
- `IsActive` (BOOLEAN) - Client active status

#### applicationmodule
**Relevant Columns:**
- `pkApplicationModuleId` (INT, PK) - Module identifier
- `Name` (VARCHAR) - Module name (e.g., 'Chatbot')
- `IsActive` (BOOLEAN) - Module active status

#### clientsmodule
**Relevant Columns:**
- `fkClientID` (INT, FK) - References client.pkClientId
- `fkApplicationModuleID` (INT, FK) - References applicationmodule.pkApplicationModuleId
- `IsAccess` (BOOLEAN) - Client has access to module

---

## Authentication Flow Diagrams

### Flow 1: User Query Request

```
┌─────────────┐
│   Frontend  │
│   (Angular) │
└──────┬──────┘
       │ POST /api/Chatbot/GetResponse
       │ Headers: { token: "user-session-token" }
       │ Query: ?usermsg="What is my leave balance?"
       ▼
┌──────────────────────────────────────────────────────┐
│ ChatbotController.GetResponse()                      │
│                                                       │
│ 1. Extract token from headers                        │
│ 2. Get UserData from memCache                        │
│    → ClientID, UserName                              │
│                                                       │
│ 3. Call ChatbotBusiness.GetClientInfo(clientId)      │
│    ┌─────────────────────────────────────────────┐  │
│    │ ChatbotBusiness.GetClientInfo()             │  │
│    │                                             │  │
│    │ Call ChatbotRepository.GetClientInfo()      │  │
│    │ ┌───────────────────────────────────────┐  │  │
│    │ │ ChatbotRepository.GetClientInfo()     │  │  │
│    │ │                                       │  │  │
│    │ │ Execute: usp_Chatbot_GetClientAPIKey │  │  │
│    │ │ ┌─────────────────────────────────┐  │  │  │
│    │ │ │ MySQL Database                  │  │  │  │
│    │ │ │                                 │  │  │  │
│    │ │ │ 1. Find Chatbot module ID       │  │  │  │
│    │ │ │ 2. Validate client access       │  │  │  │
│    │ │ │ 3. Return ClientInfo            │  │  │  │
│    │ │ └─────────────────────────────────┘  │  │  │
│    │ │ Return: ClientInfo object            │  │  │
│    │ └───────────────────────────────────────┘  │  │
│    │ Return: ClientInfo                          │  │
│    └─────────────────────────────────────────────┘  │
│                                                       │
│ 4. Validate API key exists                           │
│    if (apiKey == null) → 403 Forbidden               │
│                                                       │
│ 5. Call ChatbotAPI(msg, clientId, pin, key, user)    │
│    ┌─────────────────────────────────────────────┐  │
│    │ POST to FastAPI                             │  │
│    │ URL: {FastAPIURL}/query                     │  │
│    │ Headers:                                    │  │
│    │   X-Client-ID: 1                            │  │
│    │   X-Company-Pin: "1032"                     │  │
│    │   X-API-Key: "FLOW-1-abc..."                │  │
│    │   X-User-Name: "admin"                      │  │
│    │ Body: { "query": "What is my leave..." }    │  │
│    └─────────────────────────────────────────────┘  │
│                                                       │
│ 6. Convert Markdown response to HTML                 │
│ 7. Return formatted response                         │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Frontend  │
│   Displays  │
│   Response  │
└─────────────┘
```

---

### Flow 2: API Key Generation

```
┌─────────────┐
│   Admin UI  │
│  (Generate  │
│   API Key)  │
└──────┬──────┘
       │ POST /api/Admin/GenerateAPIKey
       │ Body: { clientId: 1, createdBy: 100 }
       ▼
┌──────────────────────────────────────────────────────┐
│ ChatbotBusiness.GenerateAPIKeyForClient()            │
│                                                       │
│ 1. Generate Secure Key                               │
│    ┌─────────────────────────────────────────────┐  │
│    │ GenerateSecureAPIKey(clientId)              │  │
│    │                                             │  │
│    │ message = "1-638123456789012345"            │  │
│    │ secretKey = "FlowHCM-Secret-Key-2025"       │  │
│    │ hash = HMACSHA256(message, secretKey)       │  │
│    │ key = Base64(hash).Clean().Substring(0,40)  │  │
│    │                                             │  │
│    │ Return: "FLOW-1-a3f8d9e2c1b4567890..."      │  │
│    └─────────────────────────────────────────────┘  │
│                                                       │
│ 2. Save to MySQL                                     │
│    ┌─────────────────────────────────────────────┐  │
│    │ ChatbotRepository.GenerateAPIKey()          │  │
│    │                                             │  │
│    │ Execute: usp_Chatbot_GenerateAPIKey         │  │
│    │ ┌───────────────────────────────────────┐  │  │
│    │ │ MySQL Database                        │  │  │
│    │ │                                       │  │  │
│    │ │ UPDATE client                         │  │  │
│    │ │ SET api_key = "FLOW-1-abc..."         │  │  │
│    │ │ WHERE pkClientId = 1                  │  │  │
│    │ │                                       │  │  │
│    │ │ Return: "FLOW-1-abc..."               │  │  │
│    │ └───────────────────────────────────────┘  │  │
│    └─────────────────────────────────────────────┘  │
│                                                       │
│ 3. Get Client Info                                   │
│    clientInfo = GetClientInfo(1)                     │
│    companyPin = "1032"                               │
│                                                       │
│ 4. Sync to PostgreSQL (Async)                        │
│    ┌─────────────────────────────────────────────┐  │
│    │ Task.Run(SyncToPostgreSQL())                │  │
│    │                                             │  │
│    │ POST to FastAPI                             │  │
│    │ URL: {FastAPIURL}/admin/sync-client         │  │
│    │ Body: {                                     │  │
│    │   client_id: 1,                             │  │
│    │   company_pin: "1032",                      │  │
│    │   api_key: "FLOW-1-abc...",                 │  │
│    │   is_active: true                           │  │
│    │ }                                           │  │
│    │ ┌───────────────────────────────────────┐  │  │
│    │ │ FastAPI                               │  │  │
│    │ │                                       │  │  │
│    │ │ Save to PostgreSQL                    │  │  │
│    │ │ ┌─────────────────────────────────┐  │  │  │
│    │ │ │ PostgreSQL Database             │  │  │  │
│    │ │ │                                 │  │  │  │
│    │ │ │ INSERT/UPDATE clients table     │  │  │  │
│    │ │ └─────────────────────────────────┘  │  │  │
│    │ └───────────────────────────────────────┘  │  │
│    └─────────────────────────────────────────────┘  │
│    (Fire-and-forget, doesn't block)                  │
│                                                       │
│ 5. Return API Key                                    │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Admin UI  │
│   Displays  │
│   API Key   │
└─────────────┘
```

---

## Configuration Requirements

### Web.config (HCMSAPI)

```xml
<configuration>
  <connectionStrings>
    <add name="conString" 
         connectionString="Server=localhost;Database=flowhcm;Uid=root;Pwd=password;" 
         providerName="MySql.Data.MySqlClient" />
  </connectionStrings>
  
  <appSettings>
    <!-- FastAPI Chatbot Service URL -->
    <add key="FastAPIURL" value="http://localhost:8000" />
    
    <!-- Secret key for API key generation (HMACSHA256) -->
    <add key="APIKeySecret" value="FlowHCM-Secret-Key-2025" />
  </appSettings>
</configuration>
```

**Required Settings:**
- `conString` - MySQL connection string for FlowHCM database
- `FastAPIURL` - Base URL of FastAPI chatbot service
- `APIKeySecret` - Secret key for HMACSHA256 hashing (optional, has default)

---

## Security Considerations

### 1. API Key Generation
- Uses HMACSHA256 for cryptographic security
- Includes timestamp to ensure uniqueness
- 40-character random string
- Format: `FLOW-{ClientID}-{40-char-hash}`

### 2. Authentication Layers
**Layer 1: FlowHCM Session Token**
- User must be logged into FlowHCM
- Token validated via memCache
- Extracts ClientID and UserName

**Layer 2: Chatbot API Key**
- Client must have API key in database
- Chatbot module must be active
- Client must have access permission

**Layer 3: FastAPI Validation**
- Validates API key in PostgreSQL
- Checks client is active
- Validates CompanyPin matches

### 3. Access Control
**Database Level:**
- `client.IsActive = 1` - Client must be active
- `applicationmodule.IsActive = 1` - Chatbot module must be enabled
- `clientsmodule.IsAccess = 1` - Client must have explicit access

**Application Level:**
- Token validation in controller
- API key validation before forwarding
- Error messages don't expose internal details

### 4. Data Protection
- API keys stored in database (not in code)
- Sensitive data passed via headers (not URL)
- HTTPS recommended for production
- No API keys logged in plain text

---

## Error Handling

### Controller Level Errors

#### 1. Missing Token
**Status:** 400 BadRequest
**Message:** "Missing or invalid token."
**Cause:** No token in request headers

#### 2. No API Key
**Status:** 403 Forbidden
**Message:** "Chatbot access is not enabled for your organization. Please contact your administrator."
**Cause:** 
- Client has no API key in database
- Chatbot module not active
- Client doesn't have access permission

#### 3. FastAPI Error
**Status:** 400 BadRequest
**Message:** "Sorry, I'm having trouble connecting right now. Please try again in a moment."
**Cause:** FastAPI returned error response

#### 4. General Exception
**Status:** 400 BadRequest
**Message:** "We're sorry, but something went wrong. We've been notified about this issue..."
**Cause:** Unhandled exception in processing

### Business Layer Errors

#### 1. Repository Error
**Exception:** "Error getting client info: {details}"
**Cause:** Database connection or query error

#### 2. Key Generation Error
**Exception:** "Error generating API key: {details}"
**Cause:** Failed to save key or sync to PostgreSQL

#### 3. Sync Error
**Exception:** "FastAPIURL not configured in Web.config"
**Cause:** Missing configuration setting

### Repository Layer Errors
- Database connection failures
- Stored procedure not found
- Invalid parameters
- All wrapped and thrown to business layer

---

## Testing Checklist

### Prerequisites
- [ ] MySQL database with FlowHCM schema
- [ ] Stored procedures created
- [ ] Chatbot module added to `applicationmodule` table
- [ ] Client has entry in `clientsmodule` with access
- [ ] FastAPI service running
- [ ] Web.config configured with FastAPIURL

### Test Cases

#### 1. API Key Generation

- [ ] Generate API key for test client
- [ ] Verify key format: `FLOW-{ClientID}-{40-chars}`
- [ ] Verify key saved in MySQL `client.api_key`
- [ ] Verify key synced to PostgreSQL
- [ ] Verify key is unique on regeneration

#### 2. Client Access Validation
- [ ] Test with client that has API key → Success
- [ ] Test with client without API key → 403 Forbidden
- [ ] Test with inactive client → No access
- [ ] Test with inactive Chatbot module → No access
- [ ] Test with no `clientsmodule` entry → No access

#### 3. Query Flow
- [ ] Login as valid user
- [ ] Send chatbot query
- [ ] Verify token extracted correctly
- [ ] Verify ClientID and UserName retrieved
- [ ] Verify API key fetched from database
- [ ] Verify request forwarded to FastAPI with correct headers
- [ ] Verify response converted from Markdown to HTML
- [ ] Verify response returned to frontend

#### 4. Error Scenarios
- [ ] Test with missing token → 400 BadRequest
- [ ] Test with invalid token → Error
- [ ] Test with expired token → Error
- [ ] Test with FastAPI down → User-friendly error
- [ ] Test with database down → Error logged

#### 5. PostgreSQL Sync
- [ ] Generate API key
- [ ] Verify async sync doesn't block
- [ ] Check PostgreSQL for synced data
- [ ] Test sync failure (FastAPI down) → Doesn't break main flow

---

## Troubleshooting Guide

### Issue: "Chatbot access is not enabled"

**Possible Causes:**
1. Client has no API key in database
2. Chatbot module not active
3. Client doesn't have access in `clientsmodule`

**Solutions:**
```sql
-- Check if API key exists
SELECT pkClientId, api_key FROM client WHERE pkClientId = 1;

-- Check if Chatbot module is active
SELECT * FROM applicationmodule WHERE Name = 'Chatbot';

-- If not active, activate it
UPDATE applicationmodule SET IsActive = 1 WHERE Name = 'Chatbot';

-- Check client access
SELECT * FROM clientsmodule 
WHERE fkClientID = 1 
  AND fkApplicationModuleID = (SELECT pkApplicationModuleId FROM applicationmodule WHERE Name = 'Chatbot');

-- If no access, grant it
INSERT INTO clientsmodule (fkClientID, fkApplicationModuleID, IsAccess)
VALUES (1, (SELECT pkApplicationModuleId FROM applicationmodule WHERE Name = 'Chatbot'), 1);
```

---

### Issue: Stored Procedure Not Found

**Error:** "Procedure 'usp_Chatbot_GetClientAPIKey' does not exist"

**Solution:**
Run the stored procedure creation scripts in MySQL:

```sql
-- 1. Get Client API Key
DROP PROCEDURE IF EXISTS `usp_Chatbot_GetClientAPIKey`;
CREATE PROCEDURE `usp_Chatbot_GetClientAPIKey`(IN ClientID INT)
BEGIN
    DECLARE ChatbotModuleID INT;
    
    SELECT pkApplicationModuleId INTO ChatbotModuleID 
    FROM applicationmodule 
    WHERE Name = 'Chatbot' AND IsActive = 1 
    LIMIT 1;
    
    SELECT c.pkClientId AS ClientID,
           c.CompanyPin,
           c.api_key AS APIKey,
           c.IsActive
    FROM client c
    INNER JOIN clientsmodule cm ON cm.fkClientID = c.pkClientId
    WHERE c.pkClientId = ClientID
      AND c.IsActive = 1
      AND c.api_key IS NOT NULL
      AND c.api_key != ''
      AND cm.fkApplicationModuleID = ChatbotModuleID
      AND cm.IsAccess = 1;
END;

-- 2. Generate API Key
DROP PROCEDURE IF EXISTS `usp_Chatbot_GenerateAPIKey`;
CREATE PROCEDURE `usp_Chatbot_GenerateAPIKey`(
    IN ClientID INT,
    IN APIKey VARCHAR(255),
    IN CreatedBy INT
)
BEGIN
    UPDATE client 
    SET api_key = APIKey,
        ModifiedDate = NOW(),
        ModifiedBy = CreatedBy
    WHERE pkClientId = ClientID;
    
    SELECT api_key AS APIKey 
    FROM client 
    WHERE pkClientId = ClientID;
END;
```

---

### Issue: FastAPI Connection Failed

**Error:** "Sorry, I'm having trouble connecting right now"

**Possible Causes:**
1. FastAPI service not running
2. Wrong FastAPIURL in Web.config
3. Network/firewall blocking connection

**Solutions:**
```bash
# Check if FastAPI is running
curl http://localhost:8000/health

# Check Web.config
<add key="FastAPIURL" value="http://localhost:8000" />

# Test connection from server
Test-NetConnection -ComputerName localhost -Port 8000
```

---

### Issue: PostgreSQL Sync Failing

**Symptoms:** API key works but not synced to PostgreSQL

**Check:**
1. FastAPI logs for sync errors
2. PostgreSQL connection in FastAPI
3. `/admin/sync-client` endpoint exists

**Debug:**
```csharp
// Add logging in SyncToPostgreSQL method
Console.WriteLine($"Syncing client {clientId} to PostgreSQL...");
Console.WriteLine($"FastAPI URL: {fastApiUrl}");
Console.WriteLine($"Response: {await response.Content.ReadAsStringAsync()}");
```

---

### Issue: Markdown Not Converting

**Symptoms:** Response shows raw Markdown instead of HTML

**Check:**
1. Markdig NuGet package installed
2. `ConvertMarkdownToHtml()` being called
3. Exception in conversion (check logs)

**Fallback:** Returns original text if conversion fails

---

## Performance Considerations

### 1. Database Connections
- Uses `using` statements for automatic disposal
- Connections opened only when needed
- No connection pooling issues

### 2. Async Operations
- PostgreSQL sync is fire-and-forget
- Doesn't block main request flow
- Uses `Task.Run()` for background execution

### 3. Caching
- User data cached in memCache (FlowHCM pattern)
- No repeated database lookups for same session
- Token-based cache key

### 4. HTTP Timeouts
- FastAPI requests have 30-second timeout
- Prevents hanging requests
- Graceful error handling on timeout

---

## Deployment Steps

### 1. Database Setup
```sql
-- Create Chatbot module
INSERT INTO applicationmodule (Name, IsActive, CreatedDate, CreatedBy)
VALUES ('Chatbot', 1, NOW(), 'admin');

-- Grant access to clients
INSERT INTO clientsmodule (fkClientID, fkApplicationModuleID, IsAccess)
SELECT pkClientId, 
       (SELECT pkApplicationModuleId FROM applicationmodule WHERE Name = 'Chatbot'),
       1
FROM client 
WHERE IsActive = 1;

-- Run stored procedures (see above)
```

### 2. Application Configuration
```xml
<!-- Web.config -->
<add key="FastAPIURL" value="https://chatbot-api.flowhcm.com" />
<add key="APIKeySecret" value="YOUR-PRODUCTION-SECRET-KEY" />
```

### 3. Generate API Keys
```csharp
// For each client
ChatbotBusiness business = new ChatbotBusiness();
string apiKey = business.GenerateAPIKeyForClient(clientId, adminUserId);
```

### 4. Verify FastAPI Integration
- Ensure FastAPI is deployed and accessible
- Test `/query` endpoint
- Test `/admin/sync-client` endpoint
- Verify PostgreSQL connection

### 5. Frontend Integration
```javascript
// Angular service
chatbotQuery(message: string) {
  const token = localStorage.getItem('token');
  return this.http.post(
    `${API_URL}/api/Chatbot/GetResponse?usermsg=${encodeURIComponent(message)}`,
    {},
    { headers: { token } }
  );
}
```

---

## Maintenance

### Regular Tasks
- Monitor FastAPI sync success rate
- Review error logs for failed requests
- Rotate API keys periodically (if required)
- Update `APIKeySecret` on security schedule

### Monitoring Points
- API key generation success rate
- PostgreSQL sync success rate
- FastAPI response times
- Error rate by client

### Logging
- All errors logged via `_tracer`
- Sync failures logged to console
- Client access attempts logged

---

## API Reference

### Endpoints

#### POST /api/Chatbot/GetResponse
**Description:** Send user query to chatbot

**Headers:**
- `token` (required) - User session token

**Query Parameters:**
- `usermsg` (required) - User's chat message

**Response:**
```json
{
  "Response": {
    "StatusCode": 200,
    "Message": "Valid Request",
    "Informations": [
      {
        "message": "<p>Your leave balance is 15 days.</p>"
      }
    ]
  },
  "ThrowException": false
}
```

**Error Response:**
```json
{
  "Response": {
    "StatusCode": 403,
    "Message": "Chatbot access not enabled.",
    "Errors": [
      {
        "message": "Chatbot access is not enabled for your organization..."
      }
    ]
  },
  "ThrowException": true
}
```

---

## Appendix

### A. Dependencies
- **MySql.Data** - MySQL database connectivity
- **Newtonsoft.Json** - JSON serialization
- **Markdig** - Markdown to HTML conversion
- **System.Net.Http** - HTTP client for FastAPI calls
- **System.Security.Cryptography** - HMACSHA256 for key generation

### B. Related Files
- `HCMSAPI/Controllers/ChatbotController.cs`
- `HCMS.Business/ChatbotBusiness.cs`
- `HCMS.Data/Repository/ChatbotRepository.cs`
- `HCMSAPI/Web.config`

### C. Database Scripts Location
- Stored procedures: See "Troubleshooting Guide" section
- Module setup: See "Deployment Steps" section

---

## Document Version
- **Version:** 1.0
- **Last Updated:** December 1, 2025
- **Author:** FlowHCM Development Team

---

**End of Documentation**
