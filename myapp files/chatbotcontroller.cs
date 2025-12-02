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
using Markdig;

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
                    //string formattedRes = Regex.Replace(result.ContainsKey("data") ? result["data"] : "No response", @"(?<!^)\s(?=\d+\.\s)", "\n");
                    string message = result.ContainsKey("data") ? result["data"] : "No response";

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
                        Response = new ErrorMessageResponse(HttpStatusCode.BadRequest, "Sorry, I’m having trouble connecting right now. Please try again in a moment.", errors)
                    };
                }
            }
            catch (Exception ex)
            {
                _tracer.Error(Request, this.ControllerContext.ControllerDescriptor.ControllerType.FullName, new Exception("ERROR"));
                var errors = new[] { new { message = "Sorry, I’m having trouble connecting right now. Please try again in a moment." } };
                return new MessageWithExceptionResponse { ThrowException = true, Response = new ErrorMessageResponse(HttpStatusCode.BadRequest, "We're sorry, but something went wrong. We've been notified about this issue and we'll take a look at it shortly.Contact your system admin or call/email us at +92345-1003569/support@flowhcm.com", errors) };
            }
        }
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
                var baseUrl = "http://localhost:8000"; // Replace with actual base URL
                var endpoint = "/query";
                var authToken = "abc";

                var handler = new HttpClientHandler
                {
                    UseCookies = true,
                    CookieContainer = new System.Net.CookieContainer()
                };

                using (var client = new HttpClient(handler))
                {
                    client.BaseAddress = new Uri(baseUrl);
                    //client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", authToken);
                    //clientDefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));


                    var sessionCookie = Request.Headers.GetCookies("chatbot_session").FirstOrDefault();
                    if (sessionCookie != null && sessionCookie["chatbot_session"] != null)
                    {
                        // Add existing session cookie to the request to FastAPI
                        handler.CookieContainer.Add(new Uri(baseUrl), new System.Net.Cookie("session", sessionCookie["chatbot_session"].Value));
                    }


                    var json = $"{{ \"query\": \"{msg}\" }}";
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    HttpResponseMessage response = await client.PostAsync(endpoint, content);
                    var responseContent = await response.Content.ReadAsStringAsync();

                    var result = new Dictionary<string, string>
                    {
                        ["status_code"] = response.StatusCode.ToString()
                    };

                    // Get session cookie from FastAPI response
                    var cookies = handler.CookieContainer.GetCookies(new Uri(baseUrl));
                    var sessionCookieFromResponse = cookies["session"];

                    if (sessionCookieFromResponse != null)
                    {
                        // FastAPI sent a session cookie - forward it to the browser
                        var cookie = new System.Web.HttpCookie("chatbot_session", sessionCookieFromResponse.Value)
                        {
                            HttpOnly = false,
                            Path = "/",
                            Expires = DateTime.Now.AddMinutes(5)
                        };
                        
                        // Check if cookie already exists and remove it first to avoid duplicates
                        if (HttpContext.Current.Response.Cookies["chatbot_session"] != null)
                        {
                            HttpContext.Current.Response.Cookies.Remove("chatbot_session");
                        }
                        
                        HttpContext.Current.Response.Cookies.Add(cookie);
                    }




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
                //return new Dictionary<string, string>
                //{
                //    ["status_code"] = "500",
                //    ["error"] = e.Message
                //};
            }

        }
    }
}
