$(function(){
var X = XLSX;

//http://stackoverflow.com/questions/13712697/set-background-color-in-hex
function rgbToHex(col)
{
    if(col.charAt(0)=='r')
    {
        col=col.replace('rgb(','').replace(')','').split(',');
        var r=parseInt(col[0], 10).toString(16);
        var g=parseInt(col[1], 10).toString(16);
        var b=parseInt(col[2], 10).toString(16);
        r=r.length==1?'0'+r:r; g=g.length==1?'0'+g:g; b=b.length==1?'0'+b:b;
        var colHex='FF'+r+g+b;
        return colHex;
    }
}

// http://stackoverflow.com/questions/6491463/accessing-nested-javascript-objects-with-string-key
var byString = function(o, s) {
    s = s.replace(/\[(\w+)\]/g, '.$1'); // convert indexes to properties
    s = s.replace(/^\./, '');           // strip a leading dot
    var a = s.split('.');
    for (var i = 0, n = a.length; i < n; ++i) {
        var k = a[i];
        if (k in o) {
            o = o[k];
        } else {
            return;
        }
    }
    return o;
}


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
    console.log(worksheet);

    var ref = XLSX.utils.decode_range(worksheet['!ref']);
    var max_col = ref.e.c + 1, max_row = ref.e.r + 1;


    for (z in worksheet) {
        /* all keys that do not begin with "!" correspond to cell addresses */
        if(z[0] === '!') continue;

        var cell = XLSX.utils.decode_cell(z);

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
        if (worksheet[z].w) {
            var txt = worksheet[z].w;
            txt = txt.replace(/&#10;/g,'\n'); 
            table[row][col] = txt;  
        } else {
            table[row][col] = '';
        }
    }
    console.table(table);

    var container = document.getElementById('example');
    $(container).empty();

    var merges = worksheet['!merges'];
    var mergeCells = true;
    if (merges) {
        var mergeCells = merges.map(function(x) {
            return {row: x.s.r, col: x.s.c, rowspan: x.e.r - x.s.r + 1, colspan: x.e.c - x.s.c + 1};
        });
    }
    console.table(mergeCells);

    var colWidths = worksheet['!cols'] ? worksheet['!cols'].map(function(x) {return x.wpx  }) : undefined;

    function myRenderer(instance, td, row, col, prop, value, cellProperties) {
        Handsontable.renderers.TextRenderer.apply(this, arguments);
        var cell_index = XLSX.utils.encode_cell({c:col, r:row});

        var txt = td.innerHTML;
        txt = txt.replace(/ /g,'&nbsp;');
        td.innerHTML = txt;
        var cell = worksheet[cell_index];
        if (cell) {
            if (byString(cell, 's.font.color.rgb')) td.style.color = '#' + cell.s.font.color.rgb.substring(2, 8);
            if (byString(cell, 's.font.bold')) td.style.fontWeight = cell.s.font.bold ? 'bold': 'normal';
            if (byString(cell, 's.font.italic')) td.style.fontStyle = cell.s.font.italic ? 'italic' : 'normal';
            if (byString(cell, 's.font.name')) td.style.fontFamily = cell.s.font.name;
            if (byString(cell, 's.font.sz')) td.style.fontSize = cell.s.font.sz + 'pt';
            if (byString(cell, 's.font.underline')) td.style.textDecoration = cell.s.font.underline ? 'underline' : 'none';

            if (byString(cell, 's.alignment.vertical')) td.vAlign = cell.s.alignment.vertical;
            if (byString(cell, 's.alignment.horizontal')) td.style.textAlign = cell.s.alignment.horizontal;

            if (byString(cell, 's.fill.fgColor.rgb')) td.style.backgroundColor = '#' + cell.s.fill.fgColor.rgb.substring(2, 8);

            if (byString(cell, 's.border.top.style')) td.style.borderTopWidth = cell.s.border.top.style == 'medium' ? '2px' : '1px';
            if (byString(cell, 's.border.right.style')) td.style.borderRightWidth= cell.s.border.right.style == 'medium' ? '2px' : '1px';
            if (byString(cell, 's.border.bottom.style')) td.style.borderBottomWidth = cell.s.border.bottom.style == 'medium' ? '2px' : '1px';
            if (byString(cell, 's.border.left.style')) td.style.borderLeftWidth = cell.s.border.left.style == 'medium' ? '2px' : '1px';
            if (byString(cell, 's.border.top.color.rgb')) td.style.borderTopColor = '#' + cell.s.border.top.color.rgb.substring(2, 8);
            if (byString(cell, 's.border.right.color.rgb')) td.style.borderRightColor = '#' + cell.s.border.right.color.rgb.substring(2, 8);
            if (byString(cell, 's.border.bottom.color.rgb')) td.style.borderBottomColor = '#' + cell.s.border.bottom.color.rgb.substring(2, 8);
            if (byString(cell, 's.border.left.color.rgb')) td.style.borderLeftColor= '#' + cell.s.border.left.color.rgb.substring(2.8);

        }
    }

    var hot = new Handsontable(container, {
        data: table,
        rowHeaders: true,
        colHeaders: true,
        manualColumnResize: true,
        manualRowResize: true,
        contextMenu: true,
        mergeCells: mergeCells,
        colWidths: colWidths,
        cells: function (row, col, prop) {
            var cellProperties = {};
            cellProperties.renderer = myRenderer;
            return cellProperties;
        }
        });
    window['hot'] = hot;
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
                    wb = X.read(data, {type: 'binary', cellStyles: true, cellDates: true});
                } else {
                    var arr = fixdata(data);
                    wb = X.read(btoa(arr), {type: 'base64', cellStyles: true, cellDates: true});
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


function datenum(v, date1904) {
    if(date1904) v+=1462;
    var epoch = Date.parse(v);
    return (epoch - new Date(Date.UTC(1899, 11, 30))) / (24 * 60 * 60 * 1000);
}
 
function sheet_from_array_of_arrays(data, opts) {
    var ws = {};
    var range = {s: {c:10000000, r:10000000}, e: {c:0, r:0 }};
    for(var R = 0; R != data.length; ++R) {
        for(var C = 0; C != data[R].length; ++C) {
            if(range.s.r > R) range.s.r = R;
            if(range.s.c > C) range.s.c = C;
            if(range.e.r < R) range.e.r = R;
            if(range.e.c < C) range.e.c = C;
            var cell = {v: data[R][C] };
            if(cell.v == null) continue;
            var cell_ref = XLSX.utils.encode_cell({c:C,r:R});
            
            if(typeof cell.v === 'number') cell.t = 'n';
            else if(typeof cell.v === 'boolean') cell.t = 'b';
            else if(cell.v instanceof Date) {
                cell.t = 'n'; cell.z = XLSX.SSF._table[14];
                cell.v = datenum(cell.v);
            }
            else cell.t = 's';
            
            var elem = hot.getCell(R, C, false);
            if (elem) {
                cell.s = {};
                if (byString(elem, 'style.backgroundColor')) cell.s.fill = {fgColor: {rgb: rgbToHex(elem.style.backgroundColor)}};

                cell.s.alignment = {};
                if (byString(elem, 'vAlign')) cell.s.alignment.vertical = elem.vAlign;
                if (byString(elem, 'style.textAlign')) cell.s.alignment.horizontal= elem.style.textAlign;

                cell.s.font = {};
                if (byString(elem, 'style.color')) cell.s.font.color = {rgb: rgbToHex(elem.style.color)};
                if (byString(elem, 'style.fontWeight')) cell.s.font.bold = elem.style.fontWeight == 'bold';
                if (byString(elem, 'style.fontStyle'))  cell.s.font.italic = elem.style.fontStyle== 'italic';
                if (byString(elem, 'style.fontFamily')) cell.s.font.name = elem.style.fontFamily;
                if (byString(elem, 'style.fontSize')) cell.s.font.sz = elem.style.fontSize.replace(/[^0-9]/g, '');
                if (byString(elem, 'style.textDecoration')) cell.s.font.underline = true;
            }
            
            ws[cell_ref] = cell;
        }
    }
    if(range.s.c < 10000000) ws['!ref'] = XLSX.utils.encode_range(range);
    return ws;
}

function Workbook() {
    if(!(this instanceof Workbook)) return new Workbook();
    this.SheetNames = [];
    this.Sheets = {};
}
 
function s2ab(s) {
    var buf = new ArrayBuffer(s.length);
    var view = new Uint8Array(buf);
    for (var i=0; i!=s.length; ++i) view[i] = s.charCodeAt(i) & 0xFF;
    return buf;
}

$('#btn-export').on('click', function() {
    var theTable = $('#example table')[0]; 
    var ranges = hot.mergeCells.mergedCellInfoCollection.map(function(x) {
        return {s: {r: x.row, c: x.col}, e: {r: x.row + x.rowspan - 1, c: x.col + x.colspan -1}};
    });

    /* original data */
    var data = hot.getData(); 

    var ws_name = "SheetJS";
    console.table(data); 

    var wb = new Workbook(), ws = sheet_from_array_of_arrays(data);

    /* add ranges to worksheet */
    ws['!merges'] = ranges;

    /* add worksheet to workbook */
    wb.SheetNames.push(ws_name);
    wb.Sheets[ws_name] = ws;

    var wbout = XLSX.write(wb, {bookType:'xlsx', bookSST:false, type: 'binary'});

    saveAs(new Blob([s2ab(wbout)],{type:"application/octet-stream"}), "test.xlsx")

});

});

