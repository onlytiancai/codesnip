$(function(){
var X = XLSX;



function fixdata(data) {
    var o = "", l = 0, w = 10240;
    for(; l<data.byteLength/w; ++l) o+=String.fromCharCode.apply(null,new Uint8Array(data.slice(l*w,l*w+w)));
    o+=String.fromCharCode.apply(null, new Uint8Array(data.slice(l*w)));
    return o;
}

function get_col_num(z) { return z.substring(0,1).charCodeAt() - 64; }
function get_row_num(z) { return parseInt(z.substring(1), 10); }

function process_wb(workbook) {
    var max_col = 0, max_row = 0;
    var sheet_name_list = workbook.SheetNames;

    sheet_name_list.forEach(function(y) { /* iterate through sheets */
      var worksheet = workbook.Sheets[y];
      for (z in worksheet) {
        /* all keys that do not begin with "!" correspond to cell addresses */
        if(z[0] === '!') continue;

        var col = get_col_num(z);
        if (col > max_col) max_col = col;
        var row = get_row_num(z);
        if (row > max_row) max_row = row;

        console.log(y + "!" + z + "=" + JSON.stringify(worksheet[z].v));
      }
    });

    console.log('max_col=', max_col, 'max_row=', max_row);

    var table = [];
    for (var i = 0; i < max_row; i++) {
        var row = [];
        for (var j = 0; j < max_col; j++) {
            row.push(''); 
        }
        table.push(row);
    }
    console.table(table);

    sheet_name_list.forEach(function(y) { /* iterate through sheets */
      var worksheet = workbook.Sheets[y];
      for (z in worksheet) {
        /* all keys that do not begin with "!" correspond to cell addresses */
        if(z[0] === '!') continue;

        var col = get_col_num(z) - 1;
        var row = get_row_num(z) - 1;
        table[row][col] = JSON.stringify(worksheet[z].v);
      }
    });

    console.table(table);

}

var drop = document.getElementById('drop');
function handleDrop(e) {
    e.stopPropagation();
    e.preventDefault();
    rABS = false; 
    use_worker = false;
    var files = e.dataTransfer.files;
    var f = files[0];
    {
        var reader = new FileReader();
        var name = f.name;
        reader.onload = function(e) {
            if(typeof console !== 'undefined') console.log("onload", new Date(), rABS, use_worker);
            var data = e.target.result;
            if(use_worker) {
                xw(data, process_wb);
            } else {
                var wb;
                if(rABS) {
                    wb = X.read(data, {type: 'binary'});
                } else {
                var arr = fixdata(data);
                    wb = X.read(btoa(arr), {type: 'base64'});
                }
                process_wb(wb);
            }
        };
        if(rABS) reader.readAsBinaryString(f);
        else reader.readAsArrayBuffer(f);
    }
}

function handleDragover(e) {
    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
}

if(drop.addEventListener) {
    drop.addEventListener('dragenter', handleDragover, false);
    drop.addEventListener('dragover', handleDragover, false);
    drop.addEventListener('drop', handleDrop, false);
}

});

