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
    var sheet_name_list = workbook.SheetNames;
    var worksheet = workbook.Sheets[sheet_name_list[0]];

        var max_col = 0, max_row = 0;
        var merges = worksheet['!merges'];
        console.log(merges);



        for (z in worksheet) {
            /* all keys that do not begin with "!" correspond to cell addresses */
            if(z[0] === '!') continue;

            var cell = XLSX.utils.decode_cell(z);
            if (cell.c + 1 > max_col) max_col = cell.c + 1;
            if (cell.r + 1 > max_row) max_row = cell.r + 1;

            console.log(z + "=" + JSON.stringify(worksheet[z].v));
        }

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

        for (z in worksheet) {
            if(z[0] === '!') continue;
            var cell = XLSX.utils.decode_cell(z);
            var col = cell.c;
            var row = cell.r;
            table[row][col] = worksheet[z].v;
        }
        console.table(table);

        var container = document.getElementById('example');
        $(container).empty();

        var mergeCells = merges.map(function(x) {
            return {row: x.s.r, col: x.s.c, rowspan: x.e.r - x.s.r + 1, colspan: x.e.c - x.s.c + 1};
        });

        console.table(mergeCells);

        var hot = new Handsontable(container, {
            data: table,
            rowHeaders: true,
            colHeaders: true,
            contextMenu: true,
            mergeCells: mergeCells
        });
}

var drop = document.getElementById('drop');
function handleDrop(e) {
    e.stopPropagation();
    e.preventDefault();
    rABS =  false; 
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

