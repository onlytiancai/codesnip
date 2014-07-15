#include <assert.h> 
#include <stdlib.h> 
#include <string.h> 

#include <CUnit/Basic.h>
#include <CUnit/Console.h>
#include <CUnit/CUnit.h>
#include <CUnit/TestDB.h>

// 测试库函数strlen功能是否正常
void test_string_lenth(void){
    char* test = "Hello";
    int len = strlen(test);
    CU_ASSERT_EQUAL(len,5);
}

// 创建一特test case，里面可以有多个测试 
CU_TestInfo testcase[] = {
    { "test_for_lenth:", test_string_lenth },
    CU_TEST_INFO_NULL
};

// suite初始化,
int suite_success_init(void) {
    return 0;
}

// suite 清理
int suite_success_clean(void) {
    return 0;
}

// 定义suite集, 里面可以加多个suit
CU_SuiteInfo suites[] = {
    // 以前的版本没有那两个NULL参数，新版需要加上，否则就coredump
    //{"testSuite1", suite_success_init, suite_success_clean, testcase },
    {"testSuite1", suite_success_init, suite_success_clean, NULL, NULL, testcase },
    CU_SUITE_INFO_NULL
};

// 添加测试集, 固定套路
void AddTests(){
    assert(NULL != CU_get_registry());
    assert(!CU_is_test_running());

    if(CUE_SUCCESS != CU_register_suites(suites)){
        exit(EXIT_FAILURE);
    }
}

int RunTest(){
    if(CU_initialize_registry()){
        fprintf(stderr, " Initialization of Test Registry failed. ");
        exit(EXIT_FAILURE);
    }else{
        AddTests();
        
        // 第一种：直接输出测试结果
        CU_basic_set_mode(CU_BRM_VERBOSE);
        CU_basic_run_tests();

        // 第二种：交互式的输出测试结果
        // CU_console_run_tests();

        // 第三种：自动生成xml,xlst等文件
        //CU_set_output_filename("TestMax");
        //CU_list_tests_to_file();
        //CU_automated_run_tests();

        CU_cleanup_registry();

        return CU_get_error();

    }

}

int main(int argc, char* argv[]) {
    return  RunTest();
}
