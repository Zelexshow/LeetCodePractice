package doublePointer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/***
 * 双指针法相关
 */
public class DoublePointers {

    /***
     * 中等--11、盛水最多的容器
     */
    public int maxArea(int[] height){
        int maxArea=0,l=0,r=height.length-1;
        while(l<r){
            maxArea=Math.max(maxArea,Math.min(height[l],height[r])*(r-l));
            if (height[l]<height[r]){
                l++;
            }else{
                r--;
            }
        }
        return maxArea;
    }


    /****
     * 中等--15、三数之和
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if(nums == null || nums.length < 3) return res;
        //首先排序
        Arrays.sort(nums);

        for (int i=0;i<nums.length;i++){
            if (nums[i]>0) break;//如果第一个数都大于0了，后面的就都不要判断了
            if (i>0 && nums[i] == nums[i-1]) continue;//去重
            int L=i+1;
            int R=nums.length-1;
            while(L<R){
                int sum=nums[i]+nums[L]+nums[R];
                if (sum == 0){
                    res.add(Arrays.asList(nums[i],nums[L],nums[R]));
                    while (L<R && nums[L] == nums[L+1]) L++;//去重
                    while (L<R && nums[R] == nums[R-1]) R--;//去重
                    //正常移动指针
                    L++;
                    R--;
                }else if (sum<0){
                    L++;//递增
                }else R--;//递减
            }
        }
        return res;
    }
}
