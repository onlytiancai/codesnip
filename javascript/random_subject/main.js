var api_url = 'http://dev.techaction.cn/api'; 
var subject_types = [{type: '单选', num: 10}, {type: '多选', num: 10}];

var promises = [];
subject_types.forEach(function(type) {
	var dfd = jQuery.Deferred();
    promises.push(dfd);

	$.getJSON(api_url+'/get_subject_list?type='+type.type, function(rsp) {
		console.log('题目列表：', type.type, rsp.subjects.length, rsp.subjects);
		var groups = group_by_point(rsp.subjects);
		var subjects = random_subject(groups, type.num);
		$.post(api_url + '/save_subjects?type=' + type.type, {subjects: subjects}, function() {
			console.log('保存题型:', type);
			dfd.resolve();
		})
	})
});

$.when.apply($, promises).then(function() {
	console.log('所有题型已保存');
});

/** 随机 num 个题目, 覆盖尽量多的知识点
 *
 * groups: {a: [1,2,3], b:[4,5,6]}
 * num: int
 *
 * */
function random_subject(groups, num) {
    var ret = [];

    // 取够 ret 或取空 group
    while(ret.length < num && !$.isEmptyObject(groups)) {
        // 取出 point 列表, 并随机取出一个 point
        var points = Object.keys(groups);                             // {a: [1, 2, 3], b: [4, 5, 6]} => [a, b] 
        var random_index = Math.floor(Math.random() * points.length); // [a, b] => 1
        var point = points[random_index];                             // ([a, b], 1) => b

        // 从该 point 对应的 subjects 里随机出去一个 subject 
        var subjects = groups[point];                                 // ({a: [1, 2, 3], b: [4, 5, 6]}, b) = > [4, 5, 6]
        random_index = Math.floor(Math.random() * subjects.length);   // [4, 5, 6] => 2
        ret.push(subjects[random_index]);                             // [] => [6]

        // 移除已经取过的 subject, 以及空的 group
        subjects.splice(random_index, 1);                             // [4, 5, 6], 2 => [4, 5]
        if (subjects.length == 0) {
            delete groups[point]                                      // {a: [1, 2, 3], b: []} => {a: [1, 2, 3] }
        }
    }

    console.log('随机抽取题目:', num, ret.length, ret);
    return ret; 
}

// 按知识点分组
function group_by_point(list) {
    var ret = {};
    list.forEach(function(item) {
        var point = item.point;
        if (ret[point] == undefined) {
            ret[item.point] = [];
        }
        ret[point].push(item);
    });

    console.log('按知识点分组:', ret)
    return ret;
}
