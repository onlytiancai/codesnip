define(function(require, exports, module) {
    var splitText = function(line) {
        if (!line){
            return [];
        }
        var result = [];
        var sb = ''; 
        // 解析标志位：
        //  0: 正在解析字符
        //  1: 遇到左边的单引号
        //  2: 遇到左边的双引号
        //  3: 左单引号里遇到\
        //  4: 左双引号里遇到\
        var state = 0;
 
        for(var i in line)
        {
            var ch = line[i];
            switch (ch)
            {
                case ' ':
                case '\t':
                    switch (state)
                    {
                        case 0:
                            if (sb.length > 0)
                            {
                                result.push(sb);
                                sb = "";
                            }
                            break;
                        case 1:
                        case 2:
                            sb += ch;
                            break;
                        case 3:
                        case 4:
                            sb += '\\';
                            sb += ch;
                            break;
                    }
                    break;
                case '\'':
                    switch (state)
                    {
                        case 0:
                            state = 1;
                            break;
                        case 1:
                            state = 0;
                            result.push(sb);
                            sb = "";
                            break;
                        case 2:
                            sb += ch;
                            break;
                        case 3:
                            sb += '\'';
                            state = 1;
                            break;
                        case 4:
                            sb += '\\';
                            sb += ch;
                            state = 2;
                            break;
                    }
                    break;
                case '\"':
                    switch (state)
                    {
                        case 0:
                            state = 2;
                            break;
                        case 1:
                            sb += ch;
                            break;
                        case 2:
                            state = 0;
                            result.push(sb);
                            sb = '';
                            break;
                        case 3:
                            sb += '\\';
                            sb += ch;
                            state = 1;
                            break;
                        case 4:
                            sb += '\"';
                            state = 2;
                            break;
                    }
                    break;
                case '\\':
                    switch (state)
                    {
                        case 0:
                            sb += ch;
                            break;
                        case 1:
                            state = 3;
                            break;
                        case 2:
                            state = 4;
                            break;
                        case 3:
                            sb += '\\';
                            sb += ch;
                            break;
                        case 4:
                            sb += '\\';
                            sb += ch;
                            break;
                    }
                    break;
                default:
                    sb += ch;
                    break;
            }
        }

        if (sb.length > 0)
        {
            result.push(sb);
        }
        return result;
    };

    var parse_txt = function(input){
        var lines = input.split('\n');
        var result = [];
        var j = 1;

        for(var i in lines){
            var line = lines[i];
            var arr = splitText(line);
            if (arr.length != 6) continue;
            var record = {
                seq: j ++,
                sub_domain: arr[0],
                record_type: arr[1],
                record_line: arr[2],
                value: arr[3],
                mx: arr[4],
                ttl: arr[5],
                state: '等待导入'
            };
            result.push(record);
        }

        return result;
    };


    module.exports = {parse_txt: parse_txt};
});
