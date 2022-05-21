import java.util.Arrays;

public class ArraysTest {
    public static void main(String[] args) {
        int[][] nums = {{1,2},{3,4}};
        System.out.println(Arrays.deepToString(nums));       

        int a[] = new int[] { 18, 62, 68, 82, 65, 9 };
        Arrays.sort(a);
        System.out.println(Arrays.toString(a));
        System.out.println("数字 62出现的位置:"+Arrays.binarySearch(a, 62));
    } 
}
