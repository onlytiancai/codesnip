// 显示警告条
exports.showWarning = function (message) {
    const html = `
    <div class="alert alert-warning alert-dismissible" role="alert">
        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
            <span aria-hidden="true">&times;</span>
        </button>
        ${message}
    </div>
    `;
    $('#message-box').append(html);
}

exports.confirm = function (msg) {
    return window.confirm(msg);
}