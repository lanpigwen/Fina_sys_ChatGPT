let sessionId = null;
// 将对话框滚动到底部
function scrollToBottom() {
    var messages = document.getElementById("messages");
    messages.scrollTop = messages.scrollHeight;
}
// 发送消息
function sendMessage() {

var message = document.getElementById("message").value;
var messageShow=message.replace(/\n/g,"<br>");
if (message.trim() == "") {
    return;
}
// 在对话框中显示发送的消息
var messages = document.getElementById("messages");
var messageSent = document.createElement("div");
messageSent.innerHTML = messageShow;
messageSent.className = "message message-sent";
messages.appendChild(messageSent);
scrollToBottom();
// 清空输入框
document.getElementById("message").value = "";
// 向后端发送请求
$.ajax({
    type: "POST",
    url: "/chat",
    contentType: "application/json",
    data: JSON.stringify({"message": message,"session_id":sessionId}),
    success: function(data) {
        // 在对话框中显示接收的消息
        var messageReceived = document.createElement("div");
        messageReceived.innerHTML = data["message"];
        messageReceived.className = "message message-received";
        messages.appendChild(messageReceived);
        sessionId=data["session_id"];
        scrollToBottom();
    },
    error: function(error) {
        console.log(error);
    }
});
}

// 发送按钮点击事件
document.getElementById("send").addEventListener("click", function() {
sendMessage();
});

// 回车键按下事件
document.getElementById("message").addEventListener("keydown", function(event) {
if (event.keyCode == 13) {
    sendMessage();
}
});
