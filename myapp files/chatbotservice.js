(function () {
    'use strict';

    angular
        .module('coreModule')
        .factory('chatbotService', chatbotService);
        //.factory('chatbotService', chatbotService);

    chatbotService.$inject = ['$http', 'reqHeader'];

    function chatbotService($http, reqHeader) {

        var service = {
            GetMsgResponse: GetMsgResponse
        }
        return service;

        function GetMsgResponse(usermsg, callback) {

            var completeURL = reqHeader.URL + "/Chatbot/GetResponse";

            var req = {
                method: serviceConfig.methodType.POST.value,
                url: completeURL,
                headers: reqHeader.secureHeader(reqHeader.contentType.Content_json),
                params: {
                    usermsg: usermsg
                }
            }

            return $http(req)
            .then(function successCallback(response) {
                if (response.statusText == "OK" || response.status == 200) {
                    var responseData = { success: true, message: response.data.Response.message, botresponse: response.data.Response};
                }
                callback(responseData);

            }, function errorCallback(response) {

                if (response.status == 401) {response.statusText ="Unauthorized"
                    var responseData = { success: false, message: response.statusText, exception: response.data.message };
                }
                else if (response.status == 500) {response.statusText = "Internal Server Error"
                    var responseData = { success: false, message: response.statusText, exception: response.data.message };
                }
                else if (response.status == 400) {response.statusText = "Bad Request"
                    var responseData = { success: false, message: response.statusText, exception: response.data.message };
                }
                else {
                    var responseData = { success: false, message: response.statusText, exception: "Please check your internet connectivity." };
                }
                callback(responseData);
            });
        }

    }
})();