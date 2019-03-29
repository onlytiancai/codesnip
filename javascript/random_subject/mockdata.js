Mock.mock(/http:\/\/dev.techaction.cn\/api\/get_subject_list.*/, {
    'subjects|5-20': [{
        'id': '@integer(1000, 9999)',
        'point|1': ['a', 'b', 'c', 'd'] 
    }]
});
Mock.mock(/http:\/\/dev.techaction.cn\/api\/save_subjects.*/, { });
