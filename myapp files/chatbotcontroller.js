define(function (require) {

  var controllerId = 'chatbotController';

  angular
    .module('coreModule')
    .controller(controllerId, chatbotController);

  chatbotController.$inject = ['$rootScope', '$location', '$scope', 'common', 'chatbotService', '$sce'];


  function chatbotController($rootScope, $location, $scope, common, chatbotService, $sce) {
    $scope.isOpen = false;
    $scope.messages = [];
    $scope.userInput = '';
    var currentMsg = '';

    $scope.toggleChat = function () {
      $scope.isOpen = !$scope.isOpen;
    };
  
    $scope.sendMessage = function () {
        if (!$scope.userInput.trim()) return;
        $scope.messages.push({ sender: 'user', text: $scope.userInput, isTyping: false, timestamp: new Date() });
        currentMsg = $scope.userInput;
        $scope.userInput = '';

        $scope.messages.push({ sender: 'bot', isTyping: true});
        var typingIndex = $scope.messages.length - 1;

        setTimeout(() => {

           
            chatbotService.GetMsgResponse(currentMsg, function (response) {
                var reply = "Sorry, I didn't understand thats"
                let botMsg = { sender: 'bot', isTyping: false};
                if (response.success) {
                    if (response.botresponse.informations) {
                        reply = response.botresponse.informations[0].message;
                    } else {
                        reply = response.botresponse.errors[0].message;
                    }
                }
                else {
                    reply = response.message;
                }
                botMsg.text = $sce.trustAsHtml(reply);
                botMsg.timestamp = new Date();
                $scope.messages[typingIndex] = botMsg;
            $scope.$apply(); // only if needed
        });
        }, 400);
    };

    $scope.checkEnter = function (event) {
      if (event.keyCode === 13) {
        $scope.sendMessage();
      }
    };

  };
});
