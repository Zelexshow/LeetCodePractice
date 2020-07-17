package dp;

import java.util.*;

/***
 * 动态规划
 */
public class DP {
    /***
     * 中等--300、最长上升子序列，
     */
    public int LengthOfLIS(int[] nums){
        if (nums == null || nums.length == 0) return 0;
        //dp[i]表示以i结尾的最长上升子序列的长度
        int[] dp=new int[nums.length];
        Arrays.fill(dp,1);
        int res=1;
        for (int i=1;i<nums.length;i++){
            for (int j=0;j<i;j++){
                if (nums[j]<nums[i]) dp[i]=Math.max(dp[i],dp[j]+1);
            }
            res=Math.max(dp[i],res);
        }
        return res;
    }
    /***
     * 困难--42、接雨水，按列计算+动规
     */
    public int trap(int[] height){
        //使用两个数组max_left[i] max_right[i]分别代表第i列左边最高的墙的高度，第i列右边边最高的墙的高度
        int sum=0;
        int[] max_left=new int[height.length];
        int[] max_right=new int[height.length];
        for (int i=1;i<max_left.length;i++) max_left[i]=max_left[i-1]<height[i-1]?height[i-1]:max_left[i-1];
        for (int j=height.length-2;j>=0;j--) max_right[j]=max_right[j+1]<height[j+1]?height[j+1]:max_right[j+1];
        for (int i=1;i<height.length-1;i++){//按列开始计算雨量，最左列和最右列不需要计算
            int less=max_left[i]<max_right[i]?max_left[i]:max_right[i];
            if (less>height[i]) sum+=less-height[i];
        }
        return sum;
    }
    /***
     * 中等--139、单词拆分
     * 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，
     * 判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        //dp[i]表示索引为i的前面的字串是否在字典里
        boolean[] dp=new boolean[s.length()+1];
        dp[0]=true;//空集当然在字典里
        for (int i=1;i<=s.length();i++){
            for (int j=0;j<i;j++){
                if (dp[j] && wordDict.contains(s.substring(j,i))){
                    dp[i]=true;
                    break;//只要匹配上了就不用再继续分了
                }
            }
        }
        return dp[s.length()];
    }
    /***
     * 中等--322、零钱兑换
     */
    public int coinChange(int[] coins,int amount){
        if (coins.length == 0) return -1;
        //声明一个amount+1长度的数组dp，代表各个价值的钱包，第0个钱包可以容纳的总价值为0，其它全部初始化为无穷大
        //dp[i]表示当钱包的总价值为i时，所需的最小硬币数
        int[] dp=new int[amount+1];
        Arrays.fill(dp,1,dp.length,Integer.MAX_VALUE);
        //dp[0]=0
        for (int money=1;money<=amount;money++){
            //对所有硬币进行挑选
            for (int i=0;i<coins.length;i++){
                //只用当money大于等于coins[i]时，才有可能使用coins[i]进行兑换
                if (money>=coins[i] && dp[money-coins[i]] !=Integer.MAX_VALUE){
                    dp[money]=Math.min(dp[money],dp[money-coins[i]]+1);
                }
            }
        }
        if (dp[amount]!=Integer.MAX_VALUE) return dp[amount];
        return -1;
    }

    //二刷
    public int coinChange1_2(int[] coins,int amount){
        if (coins == null || coins.length == 0) return -1;
        //本质上是0-1背包
        int[] dp=new int[amount+1];//dp[i]表示金额为i的时候最小的钱币数
        Arrays.fill(dp,1,dp.length,Integer.MAX_VALUE);
        for (int i=0;i<=amount;i++){
            for (int j=0;j<coins.length;j++){
                if (i-coins[j]>=0 && dp[i-coins[j]] != Integer.MAX_VALUE){
                    dp[i]=Math.min(dp[i],dp[i-coins[j]]+1);
                }

            }
        }
        return dp[amount]!=Integer.MAX_VALUE?dp[amount]:-1;
    }
    /***
     * 中等--518、零钱兑换2
     */
    public int change(int amount,int[] coins){
        //完全背包问题
        //dp[i][j]:coins[0..i]范围内的硬币，组成目标金额为j,能够得到的组合数
        int len=coins.length;
        if (len == 0){
            if (amount == 0) return 1;
            else return 0;
        }
        int[][] dp=new int[len][amount+1];
        dp[0][0]=1;
        for (int i=coins[0];i<=amount;i+=coins[0]){dp[0][i]=1;}//第一行只有整数倍的时候才有1；
        for (int i=1;i<len;i++){
            for (int j=0;j<=amount;j++){
                for (int k=0;j-k*coins[i]>=0;k++){
                    dp[i][j]+=dp[i-1][j-k*coins[i]];
                }
            }
        }
        return dp[len-1][amount];

    }
    /***
     * 中等--62、不同的路径
     */
    public int uniquePaths(int m,int n){
        int[][] dp=new int[m][n];
        //填充最右边的列
        for (int i=m-1;i>=0;i--) dp[i][n-1]=1;
        //填充最下面的边
        for (int i=n-1;i>=0;i--) dp[m-1][i]=1;
        //填充中间的列
        for (int i=m-2;i>=0;i--){
            for (int j=n-2;j>=0;j--){
                dp[i][j]=dp[i+1][j]+dp[i][j+1];
            }
        }
        return dp[0][0];
    }
    /***
     * 中等--63、不同的路径2,在上题的基础上增加了障碍物
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int rows=obstacleGrid.length-1;
        int cols=obstacleGrid[0].length-1;
        int[][] dp=new int[rows+1][cols+1];

        for (int i=rows;i>=0;i--){
            if (obstacleGrid[i][cols] == 1) break;//是障碍就不考虑
            dp[i][cols]=1;
        }

        for (int i=cols;i>=0;i--){
            if (obstacleGrid[rows][i] == 1) break;//是障碍就不考虑
            dp[rows][i]=1;
        }
        for (int i=rows-1;i>=0;i--){
            for (int j=cols-1;j>=0;j--){
                if (obstacleGrid[i][j] == 1) break;
                dp[i][j]=dp[i+1][j]+dp[i][j+1];
            }
        }
        return dp[0][0];

    }

    /***
     * 中等--64、最小路径和
     */
    public int minPathSum(int[][] grid) {
        int rows=grid.length,cols=grid.length;
        int[][] dp=new int[rows][cols];
        dp[rows-1][cols-1]=grid[rows-1][cols-1];

        for (int i=rows-2;i>=0;i--) dp[i][cols-1]=dp[i+1][cols-1]+grid[i][cols-1];
        for (int j=cols-2;j>=0;j--) dp[rows-1][j]=dp[rows-1][j+1]+grid[rows-1][j];

        for (int i=rows-2;i>=0;i--){
            for (int j=cols-2;j>=0;j--){
                dp[i][j]=Math.min(dp[i+1][j],dp[i][j+1])+grid[i][j];
            }
        }
        return dp[0][0];
    }
    /***
     * 中等--1143、最长公共子序列,给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
     */
    public int longestCommonSubsequence(String text1, String text2) {
        if (text1 == null || text2 == null || text1.length() == 0 || text2.length() == 0) return 0;
        //存放记忆化数组，dp[m][n]代表第一个串0...m字母和第二个串0...n之间最大公共子序列长度，

        int[][] dp=new int[text1.length()][text2.length()];
        dp[0][0]=text1.charAt(0)== text2.charAt(0)?1:0;
        for (int i=1;i<text1.length();i++) dp[i][0]=text1.charAt(i) == text2.charAt(0)?1:dp[i-1][0];
        for (int j=1;j<text2.length();j++) dp[0][j]=text1.charAt(0) == text2.charAt(j)?1:dp[0][j-1];

        for (int i=1;i<text1.length();i++){
            for (int j=1;j<text2.length();j++){
                if (text1.charAt(i) == text2.charAt(j)){
                    dp[i][j]=dp[i-1][j-1]+1;
                }else{
                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        return dp[text1.length()-1][text2.length()-1];//返回字符串全长的最大公共序列长度
    }
    /***
     * 困难--115、不同的子序列：给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。
     */
    public int numDistinct(String s, String t) {
        //dp[i][j]表示T前面i个字符串可以由S前j个字符串组成最多的个数
        int[][] dp=new int[t.length()+1][s.length()+1];
        for (int i=0;i<s.length();i++) dp[0][i]=1;
        for (int i=1;i<=t.length();i++){
            for (int j=1;j<=s.length();j++){
                if (t.charAt(i-1) == s.charAt(j-1)) dp[i][j]=dp[i-1][j-1]+dp[i][j-1];//左边和左上相加
                else dp[i][j]=dp[i][j-1];
            }
        }
        return dp[t.length()][s.length()];
    }

    /***
     * 最长公共子串
     */
    public int lcs(String str1,String str2){
        if (str1 == null || str1.length() == 0 || str2 == null || str2.length() == 0) return 0;
        //dp[i][j]表示str1以索引为i结尾的子串和str2以索引为j结尾的子串的最大匹配长度
        int[][] dp=new int[str1.length()][str2.length()];
        for (int i=0;i<str2.length();i++){
            dp[0][i]=str2.charAt(i) == str1.charAt(0)?1:0;
        }
        for (int i=0;i<str1.length();i++){
            dp[i][0]=str2.charAt(0) == str1.charAt(i)?1:0;
        }
        int max=Integer.MIN_VALUE;
        for (int i=1;i<str1.length();i++){
            for (int j=1;j<str2.length();j++){
                if (str1.charAt(i) == str2.charAt(j)) dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=0;//因为以其结尾的子串不相同，所以就没有长度
                max=Math.max(dp[i][j],max);
            }
        }
        return max;
    }

    /***
     * 连续子数组的最大和
     */
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int[] dp=new int[nums.length];//表示以nums[i]为结尾的最大子序列
        int max=Integer.MIN_VALUE;
        dp[0]=nums[0];
        for (int i=1;i<nums.length;i++){
            dp[i]=dp[i-1]>0?dp[i-1]+nums[i]:nums[i];//如果前面的和大于0就加上，反之就丢弃
            max=dp[i]>max?dp[i]:max;
        }
        return max;
    }

    /***
     * 困难--128、最长连续序列
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        Arrays.sort(nums);
        int curLen=1;//当前的连续序列的长度
        int maxLen=1;//最长连续序列的长度
        for (int i=1;i<nums.length;i++){
            if (nums[i-1]!=nums[i]){
                if (nums[i-1]+1 == nums[i]) curLen++;
                else{
                    maxLen=Math.max(curLen,maxLen);
                    curLen=1;
                }
            }
        }
        return maxLen;
    }

    /***
     * 中等--152、乘积最大的子数组
     */
    public int maxProduct(int[] nums) {
        //和连续子数组的最大和类似
        int[] dp_max=new int[nums.length+1];//表示以第i个元素结尾 的乘积最大的子数组的乘积
        int[] dp_min=new int[nums.length+1];//表示以第i个元素结尾 的乘积最小的子数组的乘积

        int max=Integer.MIN_VALUE;
        dp_max[0]=1;
        dp_min[0]=1;
        for (int i=1;i<=nums.length;i++){
            if (nums[i-1]<0){//元素如果小于0，就互换最大最小值
                int tmp=dp_min[i-1];
                dp_min[i-1]=dp_max[i-1];
                dp_max[i-1]=tmp;
            }
            dp_max[i]=Math.max(nums[i-1],dp_max[i-1]*nums[i-1]);
            dp_min[i]=Math.min(nums[i-1],dp_min[i-1]*nums[i-1]);
            max=max>dp_max[i]?max:dp_max[i];
        }
        return max;
    }
    /***
     * 中等--325、和为Sum的最长子数组
     */
    public static int maxLength(int[] arr, int k) {
        if (arr == null || arr.length == 0) return 0;
        Map<Integer, Integer> map= new HashMap<Integer, Integer>();
        int sum=0;
        int res=0;
        map.put(0,-1);//确定和为0的索引为-1；
        for(int i=0;i<arr.length;i++){
            sum+=arr[i];
            if (map.containsKey(sum-k)){//如果之前出现过这个差值，就计算目前的最大长度
                res=Math.max(res,i-map.get(sum-k));//计算长度
            }
            if (!map.containsKey(sum)){//没出现过才加入，这样就保证了数组长度最长
                map.put(sum,i);
            }
        }
        return res;
    }

    /***
     * 简单--70、爬楼梯
     */
    public int climbStairs(int n) {
        if (n<=1) return 1;
        int[] dp=new int[n+1];
        dp[0]=1;
        dp[1]=1;
        for (int i=2;i<=n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
    /***
     * 简单--198、打家劫舍
     */
    public int rob(int[] nums){
        if (nums == null || nums.length == 0 ) return 0;
        int[] dp=new int[nums.length+1];//dp[i]表示0...i-1能获得的最大利益
        dp[0]=0;
        dp[1]=nums[0];
        for (int i=2;i<=nums.length;i++) dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i-1]);
        return dp[nums.length];
    }
    /***
     * 中等--213、打家劫舍2，环形房屋，第一个和最后一个相邻
     */
    public int rob2(int[] nums) {
        int n = nums.length;
        if (n == 1) return nums[0];
        return Math.max(robRange(nums, 0, n - 2),
                robRange(nums, 1, n - 1));
    }

    // 仅计算闭区间 [start,end] 的最优结果
    int robRange(int[] nums, int start, int end) {
        int n = nums.length;
        int dp_i_1 = 0, dp_i_2 = 0;//dp_i_1表示n+1,dp_i_2表示n+2,这是自底向上的写法，和1不同
        int dp_i = 0;//表示从第i间房子抢，能抢到多少
        for (int i = end; i >= start; i--) {
            dp_i = Math.max(dp_i_1, nums[i] + dp_i_2);
            dp_i_2 = dp_i_1;
            dp_i_1 = dp_i;
        }
        return dp_i;
    }

    /**
     * 中等--打家劫舍3、树
     */
      public class TreeNode {
          int val;
          TreeNode left;
          TreeNode right;
          TreeNode(int x) { val = x; }
      }

    /**
     * 4 个孙子偷的钱 + 爷爷的钱 VS 两个儿子偷的钱 哪个组合钱多，就当做当前节点能偷的最大钱数。
     */
      public int rob(TreeNode root){//递归方法默认要选择本节点的值
          if (root == null) return 0;
          int money=root.val;
          //选择左子树的孩子
          if (root.left != null) money+=(rob(root.left.left) + rob(root.left.right));
          //选择右子树的孩子
          if (root.right != null) money+=(rob(root.right.left) + rob(root.right.right));
           return Math.max(money,rob(root.left) + rob(root.right));
      }

    /***
     * 中等--134、加油站问题
     */
    public int canCompleteCircuit(int[] gas,int[] cost){
        int n=gas.length;
        int total_tank=0;//总共的油量，从0到startpos一共还剩多少油量
        int current_tank=0;
        int startpos=0;
        for (int i=0;i<gas.length;i++){
            total_tank+=gas[i]-cost[i];
            current_tank+=gas[i]-cost[i];
            if (current_tank<0){
                startpos=i+1;
                current_tank=0;
            }
        }
        return current_tank >=0?startpos:-1;
    }

    /***
     * 困难--871、最低加油次数
     */
    public int minRefuelStops(int target,int tank,int[][] stations){
        //大顶堆
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        int nowfuel=tank,minRes=0;//nowfule是我们拥有的油，minRes是最少的加油次数
        int stationsize=stations.length,index=0;
        while(index<stationsize && stations[index][0] <= nowfuel){
            while(index<stationsize && stations[index][0] <= nowfuel){
                //如果能够到达这个油站
                pq.offer(stations[index++][1]);//将这个加油站储存的油加入队列（以便下次取出的最大的油）
            }
            if(index<stationsize){
                //当不能到达油站时，这时就加油
                while(!pq.isEmpty() && stations[index][0] > nowfuel){
                    minRes++;//加油次数加1
                    nowfuel+=pq.poll();//每次取最多的油
                }
            }
        }
        //如果此时油不够到达target,那就一直加油
        while(nowfuel < target && !pq.isEmpty()){
            minRes+=1;//加油次数自增
            nowfuel+=pq.poll();//每次取出最多的油
        }
        return nowfuel<target?-1:minRes;
    }
    /**
     * 困难--72、编辑距离
     * dp[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数
     *
     * 状态转移方程是:
     *      dp[i][j]=dp[i-1][j-1] 当word1[i] == word[j]
     *      dp[i][j]=min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])+1，
     *  其中dp[i-1][j]表示删除动作，dp[i][j-1]表示插入动作,dp[i-1][j-1]表示替换动作
     */
    public int minDistance(String word1,String word2){
        /**采用自底向上的过程*/
        int len1=word1.length(),len2=word2.length();
        int[][] dp=new int[len1+1][len2+1];
        /***首先填充第一行*/
        for (int i=1;i<=len2;i++){dp[0][i]=dp[0][i-1]+1;}//通通都要添加操作
        for (int i=1;i<=len1;i++){dp[i][0]=dp[i-1][0]+1;}
        for (int i=2;i<=len1;i++){
            for (int j=2;j<=len2;j++){
                if (word1.charAt(i-1) == word2.charAt(j-1)) dp[i][j]=dp[i-1][j-1];//相当于抵消当前未看后面的位
                else dp[i][j]=Math.min(Math.min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1;
            }
        }
        return dp[len1][len2];
    }
    //二刷
    public int minDis(String w1,String w2){
        int len1=w1.length(),len2=w2.length();
        int[][] dp=new int[len1+1][len2];
        for (int i=1;i<len1;i++) dp[i][0]=dp[i-1][0]+1;
        for (int i=1;i<=len2;i++) dp[0][i]=dp[0][i-1]+1;
        for (int i=2;i<=len1;i++){
            for (int j=2;j<=len2;j++){
                if (w1.charAt(i-1) == w2.charAt(j-1)) dp[i][j]=dp[i-1][j-1];//相当于抵消当前未看后面的位
                else dp[i][j]=Math.min(dp[i-1][j],Math.min(dp[i][j-1],dp[i-1][j-1]))+1;
            }
        }
        return dp[len1][len2];
    }
    /***
     * 中等--161、一次编辑距离
     */
    public boolean isOneEditDistance(String s, String t) {
        int l1=s.length(),l2=t.length();
        int dif=Math.abs(l1-l2);
        if (dif>1) return false;
        if (l1>l2) return isOneEditDistance(t,s);//默认s是短串
        for (int i=0;i<l1;i++){
            if (s.charAt(i) != t.charAt(i)){
                if (dif == 1) return s.substring(i).equals(t.substring(i+1));
                else return s.substring(i+1).equals(t.substring(i+1));//相当于跳过这个索引
            }
        }
        return dif == 1;//排除两串相同的情况
    }

    /***
     * 中等--91、解码方法
     */

    /**定义dp[i]是nums前i个字符可以得到的解码种数，假设之前的字符串是abcx，现在新加入了y，则有以下5种情况：
     如果x=='0'，且y=='0'，无法解码，返回0；
     如果只有x=='0'，则y只能单独放在最后，不能与x合并(不能以0开头)，此时有：
     dp[i] = dp[i-1]
     如果只有y=='0'，则y不能单独放置，必须与x合并，并且如果合并结果大于26，返回0，否则有：
     dp[i] = dp[i-2]
     如果 xy<=26: 则y可以“单独”放在abcx的每个解码结果之后后，并且如果abcx以x单独结尾，此时可以合并xy作为结尾，而这种解码种数就是abc的解码结果，此时有：
     dp[i+1] = dp[i] + dp[i-1]
     如果 xy>26: 此时x又不能与y合并，y只能单独放在dp[i]的每一种情况的最后，此时有：
     dp[i+1] = dp[i]
     */
    public int numDecodings(String s) {
        int n=s.length();
        int[] dp=new int[n+1];
        dp[0]=1;
        dp[1]=s.charAt(0) == '0'?0:1;
        if (n<=1) return dp[1];
        for (int i=2;i<=n;i++){
            int num=(s.charAt(i-2)-'0')*10+s.charAt(i-1);
            if (s.charAt(i-1) == '0' && s.charAt(i-2) == '0') return 0;//说明不能合并且不能单独存在
            else if(s.charAt(i-2) == '0') dp[i]=dp[i-1];
            else if (s.charAt(i-1) == 0){
                if (num>26) return 0;//超过26说明不存在
                dp[i]=dp[i-2];//i-1的元素和索引为i-2的合并
            }else if (num>26){ // i-1和i-2都不等于0，且n>26
                dp[i]=dp[i-1];//此时只能单独的放在i-2之后
            }else if (num<=26){//有合并和不合并两种情况
                dp[i]=dp[i-1]+dp[i-2];
            }
        }
        return dp[n];
    }

    /**
     * 困难--174、地下城游戏
     */
    public int calculateMinimumHP(int[][] dungeon) {
        int rows=dungeon.length;
        int cols=dungeon[0].length;

        int[][] dp=new int[rows][cols];//表示从i,j出发最少需要多少点数，尚未进入
        dp[rows-1][cols-1]=dungeon[rows-1][cols-1]<=0?-dungeon[rows-1][cols-1]+1:1;//最后一个格子进入前需要的最少能量
        for (int i=rows-2;i>=0;i--) {
            if (dungeon[i][cols-1] <= 0) dp[i][cols-1]=-dungeon[i][cols-1]+dp[i+1][cols-1];//小于0的情况
            else{//大于0的情况
                if (dp[i+1][cols-1]-dungeon[i][cols-1]>=1) dp[i][cols-1]=dp[i+1][cols-1]-dungeon[i][cols-1];
                else{
                    dp[i][cols-1]=1;
                }
            }
        }
        //填充最底的边
        for (int j=cols-2;j>=0;j--){
            if (dungeon[rows-1][j] <= 0){
                dp[rows-1][j]=-dungeon[rows-1][j]+dp[rows-1][j+1];
            }else{
                if (dp[rows-1][j+1]-dungeon[rows-1][j]>=1){//保证不会出现小于1的情况
                    dp[rows-1][j]=dp[rows-1][j+1]-dungeon[rows-1][j];
                }else
                    dp[rows-1][j]=1;
            }
        }
        for (int i=rows-2;i>=0;i--){
            for (int j=rows-2;j>=0;j--){
                if (dungeon[i][j]<=0){
                    dp[i][j]=-dungeon[i][j]+Math.min(dp[i+1][j],dp[i][j+1]);
                }else{
                    int tmp=Math.min(dp[i+1][j],dp[i][j+1]);
                    if (tmp-dungeon[i][j]>=1) dp[i][j]=tmp-dungeon[i][j];
                    else dp[i][j]=1;
                }
            }
        }
        return dp[0][0];
    }

    /***
     * 中等--221、最大正方形
     */
    /**dp[i][j]表示以i-1,j-1 为索引的点作为右下角的最大边长*/
    /**状态方程表示了最大边长受限于左边，上边，和左上边*/
    public int maximalSquare(char[][] matrix) {
        int r=matrix.length;
        int c=matrix[0].length;
        int[][] dp=new int[r+1][c+1];
        int maxsqlen=0;//最大的边长
        for(int i=1;i<=r;i++){
            for (int j=1;j<=c;j++){
                if (matrix[i-1][j-1] == '1'){
                    dp[i][j]=Math.min(Math.min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1])+1;
                    maxsqlen=Math.max(maxsqlen,dp[i][j]);
                }
            }
        }
        return maxsqlen*maxsqlen;
    }

    //二刷
    public int maximalSquare1_2(char[][] matrix){
        if (matrix == null || matrix.length == 0) return 0;
        int rows=matrix.length,cols=matrix[0].length;
        int max=0;
        int[][] dp=new int[rows+1][cols+1];
        for (int i=1;i<=rows;i++){
            for (int j=1;j<=cols;j++){
                if (matrix[i-1][j-1] == '1'){
                    dp[i][j]=Math.min(Math.min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1])+1;
                    max=Math.max(max,dp[i][j]);
                }
            }
        }
        return max*max;
    }
    /***
     * 简单--256、粉刷房子
     */
    /**	 三色可选
     * dp[i][j]表示刷到索引为i个房子，颜色选则j的最小开销
     * dp[i][0] = dp[i][0] + min(dp[i - 1][1], dp[i -1][2])
     * dp[i][1] = dp[i][1] + min(dp[i - 1][0], dp[i - 1][2])
     * dp[i][2] = dp[i][2] + min(dp[i - 1][0], dp[i - 1][1])　　　
     * */
    public int minCost(int[][] costs){
        int len=costs.length;//直接使用原数组节约开销
        if (costs == null || costs.length == 0) return 0;
        int[][] dp=costs;
        for (int i=1;i<len;i++){
            dp[i][0]=dp[i][0]+Math.min(dp[i-1][1],dp[i-1][2]);
            dp[i][1]=dp[i][1]+Math.min(dp[i-1][0],dp[i-1][2]);
            dp[i][2]=dp[i][2]+Math.min(dp[i-1][0],dp[i-1][1]);
        }
        return Math.min(dp[len-1][0],Math.min(dp[len-1][1],dp[len-1][2]));
    }
    /***
     * 中等--265、粉刷房子2，K色可选
     */
    public int minCost2(int[][] costs){
        int rows=costs.length,cols=costs[0].length;
        if (costs == null || costs.length == 0) return 0;
        int[][] dp=new int[rows][cols];
        int[] tmp=new int[cols];//按照数字的大小排放颜色，这样就避免遍历了
        for (int i=0;i<cols;i++)dp[0][i]=costs[0][i];
        if (rows>1){
            for (int i=1;i<rows;i++){
                for(int j=0;j<cols;j++) tmp[j]=costs[i-1][j];//分配好上一层的各个颜色的价格，准备排序
                Arrays.sort(tmp);
                for (int j=0;j<cols;j++){
                    //判断当前层的颜色是否和上层最小价格的颜色是否相同，相同就选次最小的，不同就选最小的
                    dp[i][j]=(costs[i-1][j] == tmp[0]?tmp[1]:tmp[0])+costs[i][j];
                }
            }
        }
        int min=Integer.MAX_VALUE;
        for (int i=0;i<cols;i++){
            min=Math.min(min,dp[rows-1][i]);
        }
        return min;
    }

    /***
     * 简单--276、栅栏涂色
     */
    public int numWays(int n,int k){
        int[] dp={0,k,k*k,0};
        if (n<=2) return dp[n];
        for (int i=2;i<n;i++){
            dp[3]=(k-1)*(dp[1]+dp[2]);
            dp[1]=dp[2];
            dp[2]=dp[3];
        }
        return dp[3];
    }

    /***
     * 中等--279、完全平方数
     */
    public int numSquares(int n){
        int[] dp=new int[n+1];
        Arrays.fill(dp,Integer.MAX_VALUE);
        dp[0]=0;
        for (int i=1;i<=n;i++){
            for (int j=1;j*j<=i;j++){
                dp[i]=Math.min(dp[i],dp[i-j*j]+1);
            }
        }
        return dp[n];
    }
    /***
     * 困难--312、戳气球
     */
    public static int maxCoins4DP(int[] nums) {
        if (nums == null) return 0;
        //创建虚拟边界
        int length=nums.length;
        int[] nums2=new int[length+2];
        System.arraycopy(nums,0,nums2,1,length);
        nums2[0]=1;
        nums2[length+1]=1;
        length=nums2.length;

        //创建dp表
        length = nums2.length;
        int[][] dp = new int[length][length];
        //i为begin,j为end,k为在i、j区间划分子问题时的边界
        for (int i=length-2;i>-1;i--){
            for (int j=i+2;j<length;j++){
                //维护一个最大值；如果i、j相邻，值为0
                int max = 0;
                for (int k=i+1;k<j;k++){
                    int tmp=dp[i][k]+dp[k][j]+nums2[i]*nums2[k]*nums2[j];
                    if (tmp>max) max=tmp;
                }
                dp[i][j]=max;
            }
        }
        return dp[0][length-1];

    }

    /***
     * 中等--375、猜数字大小2
     */
    public int getMoneyAmount(int n) {
        int[][] dp=new int[n+1][n+1];
        for (int len=2;len<=n;len++){
            for (int start=1;start<=n-len+1;start++){
                int minres=Integer.MAX_VALUE;
                for (int piv=start;piv<start+len-1;piv++){
                    int res=piv+Math.max(dp[start][piv-1],dp[piv+1][start+len-1]);
                    minres=Math.min(res,minres);
                }
                dp[start][start+len-1]=minres;
            }
        }
        return dp[1][n];
    }

    /***
     * 中等--376、摆动序列
     */
    public int wiggleMaxLength(int[] nums){
        if (nums.length<2) return nums.length;
        int[] up=new int[nums.length];
        int[] down=new int[nums.length];
         for (int i=1;i<nums.length;i++){
             for (int j=0;j<i;j++){
                 if (nums[i]>nums[j]){
                     up[i]=Math.max(up[i],down[j]+1);
                 }else if (nums[i]<nums[j]){
                     down[i]=Math.max(down[i],up[j]+1);
                 }
             }
         }
        return 1 + Math.max(down[nums.length - 1], up[nums.length - 1]);
    }

    /***
     * 中等--377、组合综合4
     */
    public int combinations4(int[] nums,int target){
        int[] dp=new int[target+1];
        dp[0]=1;
        for (int i=1;i<=target;i++){
            for (int n:nums){
                if (n<=i){
                    dp[i]+=dp[i-n];
                }
            }
        }
        return dp[target];
    }

    /***
     * 简单--121、买卖股票：只限一次
     * //简单逻辑：找到波谷，然后更新
     * //如果递减就一只更新波谷，如果上升就计算最大利润
     */
    public int maxProfit(int[] prices){
        int least=Integer.MAX_VALUE;//代表波谷
        int maxprofit=0;//最大利润
        for (int i=0;i<prices.length;i++){
            if (least>prices[i]) least=prices[i];//更新波谷
            else maxprofit=maxprofit>prices[i]-least?maxprofit:prices[i]-least;
        }
        return maxprofit;
    }

    /***
     * 中等--122、买卖股票2：不限次数
     */
    public int maxProfit2(int[] prices) {
        int len=prices.length;
        if (len<2) return 0;
        int[][] dp=new int[len][2];//0表示持有现金，1表示持有股票
        dp[0][0]=0;
        dp[0][1]=-prices[0];//持有股票的话就是负利润

        for (int i=1;i<len;i++){
            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
            dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);
        }
        return dp[len-1][0];//持有现金肯定利润最大
    }
    /***
     * 困难--123、买卖股票3：只限2次交易
     //dp[i][k][j]:表示现在是第i天，至多还可以进行k次交易，状态是（持有股票或者没持有股票）

     //最终的答案应该为dp[n-1][K][0],即最后一天，最多允许 K 次交易，没有持有股票，最多获得多少利润
     //dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
     //              max(   选择 rest  ,           选择 sell      )
     //dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
     //              max(   选择 rest  ,           选择 buy         )

     //注意base case
     //dp[-1][k][0] = 0
     //解释：因为 i 是从 0 开始的，所以 i = -1 意味着还没有开始，这时候的利润当然是 0 。
     //dp[-1][k][1] = -infinity
     //解释：还没开始的时候，是不可能持有股票的，用负无穷表示这种不可能。
     //dp[i][0][0] = 0
     //解释：因为 k 是从 1 开始的，所以 k = 0 意味着根本不允许交易，这时候利润当然是 0 。
     //dp[i][0][1] = -infinity
     //解释：不允许交易的情况下，是不可能持有股票的，用负无穷表示这种不可能。
     */
    public int maxProfit3(int[] prices) {
        if (prices.length == 0) return 0;
        int max_k=2;
        int n=prices.length;
        int[][][] dp=new int[n][max_k+1][2];
        for (int i=0;i<n;i++){
            for (int k=max_k;k>=1;k--){
                if (i-1 == -1){
                    dp[i][k][0]=0;
                    dp[i][k][1]=-prices[i];
                    continue;
                }
                dp[i][k][0]=Math.max(dp[i-1][k][0],dp[i-1][k][1]+prices[i]);
                dp[i][k][1]=Math.max(dp[i-1][k][1],dp[i-1][k-1][0]-prices[i]);//认为买入的时候才算交易时机
            }
        }
        return dp[n-1][max_k][0];
    }

    /***
     * 困难--188、买卖股票4：有限次K
     */
    //任意k，注意当k>n/2时相当于任意次买卖，所以要引入inf方法
    public int maxProfit4(int k, int[] prices) {
        int len=prices.length;
        if(k>len/2){
            return maxProfit_inf(prices);
        }
        int[][][] dp=new int[len][k+1][2];//dp[i][k][j]:表示现在是第i天，至多还可以进行k次交易，状态是（持有股票或者没持有股票）
        //base Case.即构建初始条件
        for (int i=0;i<len;i++){
            dp[i][0][1]=Integer.MIN_VALUE;
            dp[i][0][0]=0;
        }
        for (int i=0;i<len;i++){
            for (int nk=k;nk>=1;nk--){
                if (i == 0){//base Case
                    dp[i][nk][0]=0;
                    dp[i][nk][1]=-prices[i];
                    continue;
                }
                //以买的时候做一次交易的减少
                dp[i][nk][0]=Math.max(dp[i-1][nk][0],dp[i-1][nk][1]+prices[i]);
                dp[i][nk][1]=Math.max(dp[i-1][nk][1],dp[i-1][nk-1][0]-prices[i]);
            }
        }
        return dp[len-1][k][0];
    }
    //不限定买卖次数
    public int maxProfit_inf(int[] prices) {
        int len=prices.length;
        if (len<2) return 0;
        int[][] dp=new int[len][2];
        //第一维 i 表示索引为 i 的那一天（具有前缀性质，即考虑了之前天数的收益）能获得的最大利润；
        //第二维 j 表示索引为 i 的那一天是持有股票，还是持有现金。这里 0 表示持有现金（cash），1 表示持有股票（stock）。
        //因为不限制交易次数，除了最后一天，每一天的状态可能不变化，也可能转移；
        //写代码的时候，可以不用对最后一天单独处理，输出最后一天，状态为 0 的时候的值即可。

        dp[0][0]=0;
        dp[0][1]=-prices[0];//如果持有股票的话就是负利润

        for (int i=1;i<len;i++){
            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);//什么都不做或者卖出股票
            dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);//什么多不做或者买入股票
        }
        return dp[len-1][0];//返回的是现金
    }

    /***
     * 中等--309、买卖股票的最佳时机含冷冻期
     */
    public int maxProfitcooling(int[] prices){
        int n=prices.length;
        if (n<=1) return 0;
        int[][] dp=new int[n][2];
        dp[0][0]=0;
        dp[0][1]=-prices[0];
        for (int i=1;i<n;i++){
            if (i == 1){//base case
                dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
                dp[i][1]=Math.max(dp[i-1][1],-prices[i]);
            }else{
                dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
                dp[i][1]=Math.max(dp[i-1][1],dp[i-2][0]-prices[i]);//第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1 。
            }
        }
        return dp[n-1][0];
    }

    /***
     * 简单--303、区间和建索
     */
    class NumArray {
        // sum[i] 为 0 ~ i - 1 的和
        private int[] sums;//代表从0到指定索引的和
        public NumArray(int[] nums) {
            sums = new int[nums.length + 1];
            for (int i = 1; i <= nums.length; i++) {
                sums[i] = sums[i - 1] + nums[i - 1];
            }
        }

        public int sumRange(int i, int j) {
            return sums[j + 1] - sums[i];
        }
    }

    /***
     * 中等--413、等差子数列划分
     * dp[i]表示从0到i索引之间的新增的等差子数列的个数
     * if A[i]-A[i-1] == A[i-1]-A[i-2]
     * dp[i]=dp[i-1]+1(其中dp[i-1]+1表示添加了第i个元素之后新增的元素)
     * https://leetcode-cn.com/problems/arithmetic-slices/solution/deng-chai-shu-lie-hua-fen-by-leetcode/
     */
    public int numberOfArithmeticSlices(int[] A) {
        if (A.length<3) return 0;
        int[] dp=new int[A.length];
        int sum=0;
        for (int i=2;i<dp.length;i++){
            if (A[i]-A[i-1] == A[i-1]-A[i-2]){
                dp[i]=dp[i-1]+1;
                sum+=dp[i];
            }
        }
        return sum;
    }
    /***
     * 中等--343、整数拆分
     * 状态转移方程：F(n)=max{i*F(n-i),i*(n-i)},i=0....n-1;
     */
    public int integerBreak(int n) {
        int[] dp=new int[n+1];
        dp[1]=1;
        for (int i=2;i<=n;i++){
            for (int j=1;j<=i-1;j++){
                dp[i]=Math.max(dp[i],Math.max(j*dp[i-j],j*(i-j)));
            }
        }
        return dp[n];
    }

    /***
     * 中等--494、目标和
     * 0-1背包问题，不过是将取不取转化成正数或者负数
     * dp[i][j]表示数组从0-i的元素进行加减可以得到j的方法数量
     * 状态方程：
     * dp[ i ][ j ] = dp[ i - 1 ][ j - nums[ i ] ] + dp[ i - 1 ][ j + nums[ i ] ]
     */
    public int findTargetSumWays(int[] nums, int S) {
        int sum=0;
        for (int i=0;i<nums.length;i++){
            sum+=nums[i];
        }
        // 绝对值范围超过了sum的绝对值范围则无法得到
        if (Math.abs(S) > Math.abs(sum)) return 0;

        int len = nums.length;
        // - 0 +
        int t = sum * 2 + 1;
        int[][] dp = new int[len][t];
        // 初始化
        if (nums[0] == 0) {
            dp[0][sum] = 2;
        } else {
            dp[0][sum + nums[0]] = 1;
            dp[0][sum - nums[0]] = 1;
        }

        for (int i=1;i<len;i++){
            for (int j=0;j<t;j++){
                //边界
                int l=(j-nums[i]) >= 0?j-nums[i]:0;
                int r=(j+nums[i])<t?j+nums[i]:0;
                dp[i][j]=dp[i-1][l]+dp[i-1][r];
            }
        }
        return dp[len - 1][sum + S];
    }





}
