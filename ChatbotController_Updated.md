# Updated ChatbotController with Markdown to HTML Conversion

## Complete Code

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Net;
using System.Web.Http;
using System.Net.Http;
using System.Text;
using HCMSAPI.Models.Response;
using HCMSAPI.Models.Request;
using HCMS.Business;
using HCMS.Data;
using HCMS.Model;
using HCMS.Models.Response;
using HCMSAPI.Models.Responses;
using System.Web.Http.Tracing;
using HCMSAPI.Security;
using HCMS.Data.Repository;
using System.Data;
using System.IO;
using HCMS.Framework.Utilities;
using System.Threading.Tasks;
using System.Net.Http.Headers;
using System.Text.Json;
using Newtonsoft.Json;
using System.Text.RegularExpressions;
using Markdig; // ADD THIS - Install via NuGet: Install-Package Markdig

namespace HCMSAPI.Controllers
{
    public class ChatbotController : BaseController
    {
        private readonly ITraceWriter _tracer;

        public ChatbotController()
        {
            _tracer = GlobalConfiguration.Configuration.Services.GetTraceWriter();
        }

        [ActionName("GetResponse")]
        [AcceptVerbs("POST")]
        [GzipCompressionAttribute]
        [Authorization]
        public async Task<MessageWithExceptionResponse> GetResponse([FromUri] string usermsg)
        {
            return await GetResponseFunc(usermsg);
        }

        public async Task<MessageWithExceptionResponse> GetResponseFunc(string usermsg)
        {
            try
            {
                System.Net.Http.Headers.HttpRequestHeaders headers = Request.Headers;
                var token = headers.GetValues("token").First();

                if (string.IsNullOrEmpty(token))
                {
                    var errors = new[] { new { message = "ERROR" } };
                    return new MessageWithExceptionResponse
                    {
                        ThrowException = true,
                        Response = new ErrorMessageResponse(HttpStatusCode.BadRequest, "Missing or invalid token.", errors)
                    };
                }

                Dictionary<string, string> result = await ChatbotAPI(usermsg);

                if (result["status_code"] == "OK")
                {
                    string message = result.ContainsKey("data") ? result["data"] : "No response";
                    
                    // Convert Markdown to HTML for proper formatting
                    string htmlMessage = ConvertMarkdownToHtml(message);

                    _tracer.Info(Request, Convert.ToString((int)Common.LogType.Insert) + "||" + Convert.ToString((int)Common.Modules.Payroll), htmlMessage);
                    
                    var informations = new[] { new { message = htmlMessage } };
                    var response = new BasicMessageResponse(HttpStatusCode.OK, "Valid Request", informations);
                    return new MessageWithExceptionResponse { Response = response };
                }
                else
                {
                    string message = result.ContainsKey("error") ? result["error"]?.ToString() : "Unknown error";
                    var errors = new[] { new { message = message } };
                    _tracer.Error(Request, this.ControllerContext.ControllerDescriptor.ControllerType.FullName, new Exception("ERROR"));
                    return new MessageWithExceptionResponse
                    {
                        ThrowException = true,
                        Response = new ErrorMessageResponse(HttpStatusCode.BadRequest, "Sorry, I'm having trouble connecting right now. Please try again in a moment.", errors)
                    };
                }
            }
            catch (Exception ex)
            {
                _tracer.Error(Request, this.ControllerContext.ControllerDescriptor.ControllerType.FullName, new Exception("ERROR"));
                var errors = new[] { new { message = "Sorry, I'm having trouble connecting right now. Please try again in a moment." } };
                return new MessageWithExceptionResponse { ThrowException = true, Response = new ErrorMessageResponse(HttpStatusCode.BadRequest, "We're sorry, but something went wrong. We've been notified about this issue and we'll take a look at it shortly.Contact your system admin or call/email us at +92345-1003569/support@flowhcm.com", errors) };
            }
        }

        // NEW METHOD: Converts Markdown to HTML
        private string ConvertMarkdownToHtml(string markdown)
        {
            if (string.IsNullOrEmpty(markdown))
                return markdown;

            try
            {
                // Configure Markdig pipeline with advanced features including tables
                var pipeline = new MarkdownPipelineBuilder()
                    .UseAdvancedExtensions() // Enables tables, task lists, etc.
                    .Build();

                // Convert Markdown to HTML
                return Markdown.ToHtml(markdown, pipeline);
            }
            catch (Exception)
            {
                // If conversion fails, return original markdown
                return markdown;
            }
        }

        private async Task<Dictionary<string, string>> ChatbotAPI(string msg)
        {
            Dictionary<string, object> returnObj = new Dictionary<string, object>();
            try
            {
                var baseUrl = "http://localhost:8000";
                var endpoint = "/chat?message=" + msg;
                var authToken = "abc";

                using (var client = new HttpClient())
                {
                    client.BaseAddress = new Uri(baseUrl);
                    
                    var json = $"{{ \"message\": \"{msg}\", \"session_id\": \"session_123456789\", \"preferred_model\": null, \"force_refresh\": true, \"context\": {{ \"user_department\": \"HR\", \"user_role\": \"HR\" }} }}";
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    
                    HttpResponseMessage response = await client.PostAsync(endpoint, content);
                    var responseContent = await response.Content.ReadAsStringAsync();
                    
                    var result = new Dictionary<string, string>
                    {
                        ["status_code"] = response.StatusCode.ToString()
                    };

                    if (response.IsSuccessStatusCode)
                    {
                        Dictionary<string, object> temp = JsonConvert.DeserializeObject<Dictionary<string, object>>(responseContent);
                        result["data"] = temp["response"].ToString();
                    }
                    else
                    {
                        result["error"] = responseContent;
                    }

                    return result;
                }
            }
            catch (Exception e)
            {
                throw new Exception(e.Message);
            }
        }
    }
}
```

---

## Alternative: FastAPI Markdown to HTML Conversion (Commented Out)

If you prefer to convert Markdown to HTML in your FastAPI backend instead of C#, use this code:

```python
# In your FastAPI app (e.g., main.py or chatbot_router.py)

from fastapi import FastAPI
import markdown  # Install: pip install markdown

app = FastAPI()

@app.post("/chat")
async def chat(message: str, session_id: str, preferred_model: str = None, 
               force_refresh: bool = True, context: dict = None):
    # Your existing chatbot logic here
    chatbot_response = get_chatbot_response(message, session_id, context)
    
    # OPTION 1: Return Markdown as-is (current approach - C# converts)
    return {"response": chatbot_response}
    
    # OPTION 2: Convert to HTML in FastAPI (uncomment to use this instead)
    # html_response = markdown.markdown(
    #     chatbot_response,
    #     extensions=['tables', 'fenced_code', 'nl2br']  # Enable tables and other features
    # )
    # return {"response": html_response}
```

**Note:** If you use Option 2 (FastAPI conversion), remove the `ConvertMarkdownToHtml()` call in your C# code and just use the raw response.

---

## Explanation of Changes

### 1. **Added Markdig Library**
```csharp
using Markdig;
```
- You need to install this via NuGet Package Manager
- Run in Package Manager Console: `Install-Package Markdig`

### 2. **New Method: ConvertMarkdownToHtml()**
```csharp
private string ConvertMarkdownToHtml(string markdown)
{
    var pipeline = new MarkdownPipelineBuilder()
        .UseAdvancedExtensions()
        .Build();
    return Markdown.ToHtml(markdown, pipeline);
}
```
- Converts Markdown syntax to HTML
- `UseAdvancedExtensions()` enables:
  - **Tables** (with `|` pipes)
  - **Bold** (`**text**` → `<strong>text</strong>`)
  - **Italic** (`*text*` → `<em>text</em>`)
  - **Headers** (`# Title` → `<h1>Title</h1>`)
  - **Lists**, **links**, and more

### 3. **Applied Conversion in GetResponseFunc()**
```csharp
string message = result.ContainsKey("data") ? result["data"] : "No response";
string htmlMessage = ConvertMarkdownToHtml(message);
```
- Takes the raw Markdown response from your chatbot
- Converts it to HTML before returning to the client

### 4. **What This Fixes**
- **Bold text**: `**text**` becomes `<strong>text</strong>`
- **Tables**: Markdown tables render as proper HTML `<table>` elements
- **Headers**: `# Title` becomes `<h1>Title</h1>`
- **Lists**: Numbered and bulleted lists render correctly

### 5. **Frontend Display**
Your C# UI needs to display HTML. Options:
- **WebBrowser control**: `webBrowser1.DocumentText = htmlMessage;`
- **WebView2**: Modern alternative for displaying HTML
- **RichTextBox**: Would need additional HTML-to-RTF conversion

The API now returns properly formatted HTML that any web-based UI can render with full formatting including tables.
