package simpleMath;

import java.util.*;

/***
 * 简单数学问题
 */
public class SimpleMath {
    /***
     * 简单--1、两数之和，给定数组和目标值，求符合的数组集合--
     */
    public int[] twoSum(int[] nums, int target) {
        //使用Map来解决问题
        Map<Integer, Integer> map = new HashMap<>();//key表示数字，val表示对应索引
        for (int i=0;i<nums.length;i++) map.put(nums[i],i);
        for (int i=0;i<nums.length;i++){
            int need=target-nums[i];
            if (map.containsKey(need) && map.get(need) !=i){
                return new int[]{i,map.get(need)};
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
    /**
     * 简单--7、整数反转
     */
    public int reverse(int x) {
        //注意考虑溢出情况
        int sum=0,pop=0;
        while(x!=0){
            pop=x%10;
            x=x/10;
            if (sum>Integer.MAX_VALUE/10 || (sum == Integer.MAX_VALUE/10 && pop > 7)) return 0;//正向溢出
            if (sum<Integer.MIN_VALUE/10 || (sum == Integer.MIN_VALUE/10 && pop <-8)) return 0;//反向溢出
            sum+=sum *10+pop;
        }
        return sum;
    }

    /***
     * 简单--202、快乐数
     * 「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，
     * 也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。
     * 如果 n 是快乐数就返回 True ；不是，则返回 False 。

     */
    public boolean isHappy(int n){
        //使用快慢指针的思想
        int p=n,q=getNext(n);
        while(q != 1){
            p=getNext(p);//慢指针
            q=getNext(getNext(q));//快指针
            if (p == q) return false;
        }
        return true;
    }
    public int getNext(int x){
        int res=0;
        while(x!=0){
            res+=(x%10)*(x%10);
            x/=10;
        }
        return res;
    }

    /***
     * 简单--204、统计素数，厄拉多塞筛法
     */
    public int countPrimes(int n) {
        boolean[] isPrim=new boolean[n];
        Arrays.fill(isPrim,true);
        for (int i=2;i<n;i++){
            if (isPrim[i]) //i是素数，i的倍数就不是素数
                for (int j=i+i;j<n;j+=i) isPrim[j]=false;
        }
        int count=0;
        for(int i=0;i<n;i++) if (isPrim[i]) count++;
        return count;
    }

    /**
     * 简单--231、2的幂
     *
     */
    public boolean isPowerOfTwo(int n) {
        while(n>1){
            if (n%2 != 0) return false;//不能被2整除肯定为false
            n/=2;
        }
        return n == 1;
    }

    /**
     * 简单--263、丑数，仅包含质因数为2 3 5 的数称为丑数
     */
    public boolean isUgly(int num) {
        if (num == 0) return false;
        while(num%2 == 0) num/=2;
        while(num%3 == 0) num/=3;
        while(num%5 == 0) num/=5;
        return num == 1;
    }
    /**
     * 中等--264、丑数2，找到第n个丑数：三指针法
     */
    public int nthUglyNumber(int n) {
        if (n <= 0) return 0;
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);//放入第一个素数
        int min;
        int m2=0,m3=0,m5=0,i2=0,i3=0,i5=0;//m系表示的是当前的素数是以前一个素数乘上对应的因子的值，i系表示的是尚未使用过的索引
        while(list.size()<n){
            m2=list.get(i2)*2;
            m3=list.get(i3)*3;
            m5=list.get(i5)*5;
            min=Math.min(m2,Math.min(m3,m5));//找出三者中最小的
            list.add(min);
            //使用到了哪一个，就把哪一个作为当前需要作比较的
            if (min == m2) i2++;
            if (min == m3) i3++;
            if (min == m5) i5++;
        }
        return list.get(n-1);
    }

    /**
     * 简单--268、缺失的数字
     */
    public int missingNumber(int[] nums) {
        if (nums == null || nums.length == 0) return -1;
        int n=nums.length;
        int[] marke=new int[n+1];
        for (int i=0;i<n;i++){
            marke[nums[i]]++;
        }
        for (int i=0;i<=n;i++) if (marke[i] == 0) return i;
        return -1;
    }

    /**
     * 简单--367、有效的完全平方数
     */
    //二分法
    public boolean isPerfectSquare(int num) {
        if (num <= 1) return true;
        long left=2,right=num/2,mid,tmp;
        while(left <= right){
            mid=left+(right-left)/2;
            tmp=mid*mid;
            if (tmp == num) return true;
            if (tmp > num) right=mid-1;
            else left=mid+1;
        }
        return false;
    }

    /**
     * 简单--374、猜数字大小
     *
     * public class Solution extends GuessGame {
     *     public int guessNumber(int n) {
     *        int low = 1;
     *         int high = n;
     *         while (low <= high) {
     *             int mid = low + (high - low) / 2;
     *             int res = guess(mid);
     *             if (res == 0)
     *                 return mid;
     *             else if (res < 0)
     *                 high = mid - 1;
     *             else
     *                 low = mid + 1;
     *         }
     *         return -1;
     *     }
     * }
     */

    /**
     * 中等--398、随机数索引
     * 结论：假设当前正要读取第n个数据，则我们以1/n的概率留下该数据，否则留下前n-1个数据中的一个。
     */
    class Solution{
        private int[] nums;
        public Solution(int[] nums){
            this.nums=nums;
        }
        public int pick(int target){
            Random r = new Random();
            int n = 0;
            int index = 0;
            for(int i = 0;i < nums.length;i++)
                if(nums[i] == target){
                    //我们的目标对象中选取。
                    n++;
                    //我们以1/n的概率留下该数据
                    if(r.nextInt() % n == 0) index = i;
                }
            return index;
        }
    }
}
