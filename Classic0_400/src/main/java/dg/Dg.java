package dg;

import Array.Array;

import java.util.*;

/***
 * 本章主要围绕递归和回溯问题进行展开
 */
public class Dg {

    /***
     * 中等--17、电话号码的字母组合
     */
    private String[] letterMap={
            " ",    //0
            "",     //1
            "abc",  //2
            "def",  //3
            "ghi",  //4
            "jkl",  //5
            "mno",  //6
            "pqrs", //7
            "tuv",  //8
            "wxyz"  //9
    };
    ArrayList<String> res;
    public List<String> letterCombinations(String d){
        res=new ArrayList<>();
        if (d == null || d.length() == 0) return res;
        findCombinations(d,0,"");
        return res;
    }
    private void findCombinations(String sour,int index,String cur){
        if (index == sour.length()) res.add(cur);
        else{
            char tmp = sour.charAt(index);
            String con=letterMap[tmp-'0'];
            for (int i=0;i<con.length();i++){
                findCombinations(sour,index+1,cur+con.charAt(i));
            }
        }
    }
    /***
     * 中等--22、括号生成
     * 思路是做减法，即每使用一个括号就减一
     * 在递归的过程中做剪枝
     * 当前左右括号都有大于 0 个可以使用的时候，才产生分支；
     *
     * 产生左分支的时候，只看当前是否还有左括号可以使用；
     *
     * 产生右分支的时候，还受到左分支的限制，右边剩余可以使用的括号数量一定得在严格大于左边剩余的数量的时候，才可以产生分支；
     *
     * 在左边和右边剩余的括号数都等于 0 的时候结算
     */
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        if (n == 0) return res;
        dfs("",n,n,res);
        return res;
    }
    public void dfs(String curStr,int left,int right,List<String> res){
        if (left == 0 && right == 0){
            res.add(curStr);
            return;
        }
        if (left>right){//这种情况下是肯定不成功的
            return;
        }
        if (left>0) dfs(curStr+"(",left-1,right,res);
        if (right>0) dfs(curStr+")",left,right-1,res);
    }
    /***
     * 困难--23. 合并K个排序链表
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return mergeKLists(lists,0,lists.length-1);
    }
    public ListNode mergeKLists(ListNode[] lists,int left,int right){
        if (left >= right) return lists[left];
        int mid=left+(right-left)/2;
        ListNode l=mergeKLists(lists,left,mid);
        ListNode r = mergeKLists(lists, mid + 1, right);
        return merge(l,r);
    }
    public ListNode merge(ListNode left,ListNode right){
        if (left == null) return right;
        if (right == null) return left;
        ListNode ans = new ListNode(0);
        ListNode p=ans;
        while(left != null && right !=null){
            if (left.val <= right.val){
                p.next=left;
                left=left.next;
            }else{
                p.next=right;
                right=right.next;
            }
        }
        p.next=left == null?right:left;
        return ans.next;
    }
    /***
     * 困难 --37、解数独
     **/
    public void solveSudoku(char[][] board) {
        boolean[][] rowUsed = new boolean[9][10];
        boolean[][] colUsed = new boolean[9][10];
        boolean[][][] boxUsed = new boolean[3][3][10];

        //初始化
        for (int row=0;row<board.length;row++){
            for (int col=0;col<board[0].length;col++){
                int num=board[row][col]-'0';
                if (1<=num && num <= 9){
                    rowUsed[row][num]=true;
                    colUsed[col][num]=true;
                    boxUsed[row/3][col/3][num]=true;
                }
            }
        }
        // 递归尝试填充数组
        recusiveSolveSudoku(board, rowUsed, colUsed, boxUsed, 0, 0);
    }
    private boolean recusiveSolveSudoku(char[][]board, boolean[][]rowUsed, boolean[][]colUsed,
                                        boolean[][][]boxUsed, int row, int col){
        // 边界校验, 如果已经填充完成, 返回true, 表示一切结束
        if (col == board[0].length){
            col=0;
            row++;
            if (row == board.length) return true;
        }
        if (board[row][col] == '.'){
            //尝试填充1-9
            for (int num=1;num<=9;num++){
                boolean canUsed=!(rowUsed[row][num] || colUsed[col][num] ||
                        boxUsed[row/3][col/3][num]);//只要有一个true都不行

                if(canUsed){
                    rowUsed[row][num]=true;
                    colUsed[col][num]=true;
                    boxUsed[row/3][col/3][num]=true;

                    board[row][col]=(char)('0'+num);

                    if (recusiveSolveSudoku(board,rowUsed,colUsed,boxUsed,row,col+1)){
                        return true;
                    }
                    //回溯
                    board[row][col] = '.';

                    rowUsed[row][num] = false;
                    colUsed[col][num] = false;
                    boxUsed[row/3][col/3][num] = false;
                }
            }
        }else{
            return recusiveSolveSudoku(board,rowUsed,colUsed,boxUsed,row,col+1);
        }
        return false;

    }
    /***
     * 中等--77、组合 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
     */
    int n,k;
    List<List<Integer>> output=new LinkedList<>();//返回结果
    public List<List<Integer>> combine(int n, int k) {
        this.n=n;
        this.k=k;
        backtrack(1,new LinkedList<>());
        return output;
    }
    public void backtrack(int index, LinkedList<Integer> cur){
        if (cur.size() == k) output.add(new LinkedList<>(cur));
        else{
            for (int i=index;i<=n;i++){
                cur.add(i);
                backtrack(i+1,cur);
                cur.remove(cur.size()-1);
            }
        }
    }
    /***
     * 中等--39、组合总和
     */
    List<List<Integer>> res2;
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        res2=new ArrayList<>();
        backtrack(target,0,candidates,new LinkedList<>());
        return res2;
    }
    public void backtrack(int sum,int ix,int[] candidate,List<Integer> cur){
        if (sum < 0) return;
        if (sum == 0){
            res2.add(new LinkedList<>(cur));
            return;
        }
        for (int i=ix;i<candidate.length;i++){
            cur.add(candidate[i]);
            backtrack(sum-candidate[i],i,candidate,cur);
            cur.remove(cur.size()-1);
        }
    }
    /***
     * 中等--40、组合总和2
     */
    List<List<Integer>> res3;
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        res3=new LinkedList<>();
        if (candidates == null || candidates.length == 0) return res3;
        Arrays.sort(candidates);//首先排序，避免重复选择
        backtrack2(target,0,candidates,new LinkedList<>());
        return res3;
    }
    public void backtrack2(int sum,int start,int[] candidate,LinkedList<Integer> cur){
        if (sum < 0) return;
        if (sum == 0){
            if (!res3.contains(cur)){
                res3.add(new LinkedList<>(cur));
            }
            return;
        }
        for (int i=start;i<candidate.length;i++){
            cur.add(candidate[i]);
            backtrack2(sum-candidate[i],i+1,candidate,cur);
            cur.removeLast();//回溯
        }
    }
    /***
     * 中等--216、组合总和3
     */
    List<List<Integer>> res4;
    public List<List<Integer>> combinationSum3(int k, int n) {
        res4=new ArrayList<>();
        dfs(new HashSet<Integer>(),1,n,k);
        return res4;
    }
    public void dfs(Set<Integer> set,int index,int target,int k){
        if (set.size() == k){
            if (target == 0) res4.add(new LinkedList<>(set));
            return;
        }
        for (int i=index;i<=9;i++){
            if (!set.contains(i)){
                set.add(i);
                dfs(set,i+1,target-i,k);
                set.remove(i);
            }
        }
    }
    /***
     * 中等--46、全排列
     */
    //全排列--没有重复数字
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res=new ArrayList<>();
        if (nums == null || nums.length == 0) return res;
        dfs(nums,0,new ArrayList<Integer>(),new boolean[nums.length],res);
        return res;
    }
    public void dfs(int[] nums,int index,List<Integer> cur,boolean[] visited,List<List<Integer>> res){
        if (cur.size() == nums.length){
            res.add(new ArrayList<>(cur));
        }else{
            for (int i=0;i<nums.length;i++){
                if (!visited[i]){
                    cur.add(nums[i]);
                    visited[i]=true;
                    dfs(nums,index+1,cur,visited,res);
                    cur.remove(cur.size()-1);
                    visited[i]=false;
                }
            }
        }
    }
    /***
     * 中等--47、全排列2,可能出现重复数字
     */
    //这次采用交换法
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res=new ArrayList<>();
        if (nums == null || nums.length == 0) return res;
        dfs(0,new ArrayList<Integer>(),res,nums);
        return res;
    }
    public void dfs(int ix,List<Integer> cur,List<List<Integer>> res,int[] nums){
        if (cur.size() == nums.length){
            if (!res.contains(cur)) res.add(new LinkedList<>(cur));
            return;
        }
        for (int i=ix;i<nums.length;i++){
            swap(nums,ix,i);
            cur.add(nums[ix]);
            dfs(ix+1,cur,res,nums);
            swap(nums,ix,i);
            cur.remove(cur.size()-1);
        }
    }
    public void swap(int[] nums,int l,int r){
        int tmp=nums[l];
        nums[l]=nums[r];
        nums[r]=tmp;
    }
    /***
     * 中等--60、第K个排列
     */
    public String getPermutation(int n,int k){
        boolean[] visited=new boolean[n];
        //将n!种排列分为n组，每组有（n-1）！种排列
        return recursive(n,factorial(n-1),k,visited);
    }

    /**
     * @param n 剩余的数字个数，递减
     * @param f 每组的排列个数
     */
    private String recursive(int n,int f,int k,boolean[] visited){
        int offset=k%f;//组内偏移
        int groupIndex=k/f+(offset > 0?1:0);//第几组
        //在没有访问的数字里找到第groupIndex个数字
        int i=0;
        for (;i<visited.length && groupIndex > 0;i++){
            if(!visited[i]){
                groupIndex--;
            }
        }
        visited[i-1] = true;// 标记为已访问
        if(n - 1 > 0){
            // offset = 0 时，则取第 i 组的第 f 个排列，否则取第 i 组的第 offset 个排列
            return String.valueOf(i) + recursive(n-1, f/(n - 1), offset == 0 ? f : offset, visited);
        }else{
            // 最后一数字
            return String.valueOf(i);
        }
    }
    //求阶乘
    private int factorial(int i) {
        int res=1;
        while(i>=1){
            res*=i;
            i--;
        }
        return res;
    }
    /***
     * 中等--78、子集
     */
    int n1,k1;
    List<List<Integer>> output1 = new ArrayList();
    public List<List<Integer>> subsets(int[] nums) {
        n1=nums.length;
        for (k=0;k<n+1;k++){//每次添加尺寸为k的子集
            backtrack(0,new ArrayList<Integer>(),nums);
        }
        return output1;
    }
    public void backtrack(int first,List<Integer> cur,int[] nums){
        if (cur.size() == k){
            output1.add(new ArrayList<>(cur));
            return;
        }
        for (int i=first;i<n;i++){
            cur.add(nums[i]);
            backtrack(i+1,cur,nums);
            cur.remove(cur.size()-1);//回溯
        }
    }
    /***
     * 中等--90、子集2,数组中有重复元素
     */
    int n2,k2;
    List<List<Integer>> reslt=new ArrayList<>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        n2=nums.length;
        Arrays.sort(nums);
        for (k2=0;k2<=n2;k2++){
            baktrack(0,nums,new ArrayList<>());
        }
        return reslt;
    }
    public void baktrack(int ix,int[] nums,List<Integer> cur){
        if (cur.size() == k2){
            if (!reslt.contains(cur)) reslt.add(new ArrayList<>(cur));
            return;
        }
        for (int i=ix;i<nums.length;i++){
            cur.add(nums[ix]);
            baktrack(ix+1,nums,cur);
            cur.remove(cur.size()-1);
        }
    }
    /***
     * 中等--79、单词搜索
     */
    public boolean exist(char[][] board, String word) {
        int rows=board.length,cols=board[0].length;
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (board[i][j] == word.charAt(0) && backtrack(board,i,j,0,word)) return true;
            }
        }
        return false;
    }
    public boolean backtrack(char[][] board,int row,int col,int ix,String target){
        if (ix == target.length()) return true;
        if (row<0 || row>=board.length || col<0 || col>=board[0].length
                || board[row][col] != target.charAt(ix)) return false;
        char tmp=board[row][col];
        board[row][col]='#';//选了就标记，在这个路径中，防止后面的重复搜索
        if (backtrack(board,row+1,col,ix+1,target) || backtrack(board,row,col+1,ix+1,target)
            || backtrack(board,row-1,col,ix+1,target) || backtrack(board,row,col-1,ix+1,target))
            return true;
        else {
            board[row][col]=tmp;//回溯
            return false;
        }
    }
    /***
     * 困难--212、单词搜索2
     */
    public List<String> findWords(char[][] board, String[] words) {
        //将单词从大到小排列
        Arrays.sort(words, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o2.length()-o1.length();
            }
        });
        Trie trie=new Trie();
        List<String> res=new ArrayList<>();
        for (String word : words) {
            if (trie.startsWith(word)){
                res.add(word);
                continue;
            }
            if (exist2(board,word)){
                res.add(word);
                //加入前缀树
                trie.insert(word);
            }
        }
        return res;
    }
    //单词搜索
    public boolean exist2(char[][] board, String word) {
        boolean[][] visited=new boolean[board.length][board[0].length];
        int rows=board.length,cols=board[0].length;
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (word.charAt(0) == board[i][j] && backtrack(board,visited,word,i,j,0)){
                    return true;
                }
            }
        }
        return false;
    }
    public boolean backtrack(char[][] board,boolean[][] visited,String sb,int row,int col,int ix){
        if (ix == sb.length()) return true;
        if (row<0 || row >= board.length || col<0 || col >= board[0].length
                || board[row][col] != sb.charAt(ix) || visited[row][col]) return false;
        visited[row][col] = true;
        if (backtrack(board,visited,sb,row+1,col,ix+1) || backtrack(board,visited,sb,row,col+1,ix+1) ||
                backtrack(board,visited,sb,row-1,col,ix+1) ||backtrack(board,visited,sb,row,col-1,ix+1)){
            return true;
        }
        visited[row][col]=false;//回溯
        return false;
    }
    /***
     * 中等--695、岛屿中的最大面积
     */
    public int maxAreaOfIsland(int[][] grid) {
        int res=0,rows=grid.length,cols=grid[0].length;
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (grid[i][j]!=0){
                    res=Math.max(res,dfs(i,j,grid));
                }
            }
        }
        return res;
    }
    public int dfs(int i,int j,int[][] grid){
        if (i<0 || i>=grid.length || j<0 || j>=grid[0].length
        || grid[i][j] == 0) return 0;
        //把岛沉没
        grid[i][j]=0;
        return 1+dfs(i+1,j,grid)+dfs(i,j+1,grid)+dfs(i-1,j,grid)+dfs(i,j-1,grid);

    }

    /***
     * 中等--200、岛屿的数量
     */
    public int numIslands(char[][] grid) {
        int rows=grid.length,cols=grid[0].length;
        int count=0;//统计岛的数量
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                if (grid[i][j] == '1'){
                    count++;
                    dfs2(i,j,grid);
                }
            }
        }
        return count;
    }
    public void dfs2(int i,int j,char[][] g){
        if (i<0 || i>=g.length || j<0 || j>g[0].length || g[i][j] == '0') return;

        g[i][j] = '0';//沉岛
        dfs2(i+1,j,g);
        dfs2(i-1,j,g);
        dfs2(i,j+1,g);
        dfs2(i,j-1,g);
    }
    /***
     * 困难--51、N皇后
     */
    int[] rows;//列方向是否被攻击
    int[] mains;//主对角线上是否被攻击
    int[] secondary;//次对角线方向是否被攻击
    int n3;
    //输出
    List<List<String>> output2=new ArrayList<>();
    int[] queens;//皇后的位置：queens[i]表示第i行的皇后在的列数

    public List<List<String>> solveNQueens(int n) {
        //初始化
        rows=new int[n];
        mains=new int[2*n-1];
        secondary=new int[2*n -1];
        queens=new int[n];
        this.n=n;
        //从第一行开始求解N皇后
        backtrack(0);
        return output2;
    }
    private void backtrack(int row){
        if (row >= n) return;
        //分别尝试在row行中的每一列放置皇后
        for (int col=0;col<n;col++){
            if (!isNotUnderAttack(row,col)){
                placeQueen(row,col);
                //如果当前是最后一行，就找到解决方案
                if (row == n-1) addSolution();
                else backtrack(row+1);//在下一行放置皇后
                removeQueen(row,col);//回溯
            }
        }
    }
    private void addSolution() {
        List<String> solution = new ArrayList<>();
        for (int i=0;i<n;i++){
            int col=queens[i];
            StringBuilder sb=new StringBuilder();
            for (int j=0;j<col;j++) sb.append(".");
            sb.append("Q");
            for (int j=0;j<n-col-1;j++) sb.append(".");
            solution.add(sb.toString());
        }
        output2.add(solution);
    }
    private void placeQueen(int row,int col){
        //在row,col上放置皇后
        queens[row]=col;
        rows[col]=1;//当前位置上的列方向上已经有皇后了
        mains[row-col+n-1]=1;
    }
    //移除皇后
    private void removeQueen(int row, int col) {
        //移除row行上的皇后
        queens[row]=-1;
        //当前列上没有皇后了
        rows[col]=0;
        //当前主对角线上没皇后了
        mains[row-col+n-1]=0;
        //当前从对角线上也没有皇后
        secondary[row+col]=0;
    }
    //判断当前位置是否被其他皇后攻击
    public boolean isNotUnderAttack(int row,int col){
        // 判断的逻辑是：
        //      1. 当前位置的这一列方向没有皇后攻击
        //      2. 当前位置的主对角线方向没有皇后攻击
        //      3. 当前位置的次对角线方向没有皇后攻击
        int res= rows[col]+mains[row-col+n-1]+secondary[row+col];
        // 如果三个方向都没有攻击的话，则 res = 0，即当前位置不被任何的皇后攻击
        return res == 0;
    }
    /***
     * 中等--130、被围绕的区域
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) return;
        int m=board.length;
        int n=board[0].length;
        for (int i=0;i<m;i++){
            for (int j=0;j<n;j++){
                //从边缘o开始搜索
                boolean isEdge= i==0 || j == 0 || i==m-1 || j == n-1;
                if (isEdge && board[i][j] == '0') dfs(board,i,j);
            }
        }
        for (int i=0;i<m;i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') board[i][j]='X';
                if (board[i][j] == '#') board[i][j]='O';
            }
        }
    }
    public void dfs(char[][] board, int i, int j) {
        if (i < 0 || j < 0 || i >= board.length  || j >= board[0].length ||
                board[i][j] == 'X' || board[i][j] == '#') {
            // board[i][j] == '#' 说明已经搜索过了.
            return;
        }
        board[i][j] = '#';
        dfs(board, i - 1, j); // 上
        dfs(board, i + 1, j); // 下
        dfs(board, i, j - 1); // 左
        dfs(board, i, j + 1); // 右
    }
    /***
     * 中等--241、为运算表达式设计优先级
     */
    /**纯递归版本，根据运算符号将字符串分成两个结果*/
    public List<Integer> diffWaysToCompute(String input) {
        if(input.length() == 0) return new ArrayList<>();
        List<Integer> res=new ArrayList<>();

        int num=0;
        //考虑全是数字的情况
        int ix=0;
        while(ix<input.length() && !isOperation(input.charAt(ix))){
            num=num*10+input.charAt(ix)-'0';
            ix++;
        }
        //将全数字的情况直接返回
        if (ix == input.length()){
            res.add(num);
            return res;
        }

        for (int i=0;i<input.length();i++){
            //通过运算符将字符串分成两部分
            if (isOperation(input.charAt(i))){
                List<Integer> result1 = diffWaysToCompute(input.substring(0, i));
                List<Integer> result2 = diffWaysToCompute(input.substring(i + 1));

                //将两个结果依次运算
                for (int j=0;j<result1.size();j++){
                    for (int k=0;k<result2.size();k++){
                        char op = input.charAt(i);
                        res.add(caculate(result1.get(j),op,result2.get(k)));
                    }
                }
            }
        }
        return res;
    }
    private int caculate(int num1,char c,int num2){
        switch (c){
            case '+':
                return num1+num2;
            case '-':
                return num1-num2;
            case '*':
                return num1*num2;
        }
        return -1;
    }
    private boolean isOperation(char c){
        return c == '+' || c == '-' || c == '*';
    }
    /***
     * 中等--254、因子的组合
     */
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> results = new ArrayList<>();
        find(2, n, new ArrayList<>(), results);
        return results;
    }
    /**深搜法*/
    public void find(int from, int n, List<Integer> factors,List<List<Integer>> results){
        if (n == 1){
            if(!factors.isEmpty()){
                results.add(new ArrayList<>(factors));
            }
            return;
        }
        for (int i=from;i*i<=n;i++){
            if (n%i == 0){
                factors.add(i);
                find(i,n/i,factors,results);
                factors.remove(factors.size()-1);//回溯
            }
        }
    }

    /***
     * 困难--282、给表达式添加运算符
     */
    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        dfs(num,target,0,0,"",res);
        return res;
    }
    private void dfs(String num, int target, long cur, long preCurDiff, String item, List<String> res){
        if (cur == target && num.length() == 0){//cur加到了target 并且没有剩余的num string
            res.add(item);
            return;
        }
        //从头开始，每次从头取不同长度的string作为curStr,作为首个数字
        for (int i=1;i<=num.length();i++){
            String curStr = num.substring(0, i);
            if (curStr.length()>1 && curStr.charAt(0) == '0') break; //去掉corner case 1*05
            String nextStr = num.substring(i);
            if (item.length() == 0){//说明为第一个数字
                dfs(nextStr,target,Long.valueOf(curStr),Long.valueOf(curStr),curStr,res);
            }else{
                dfs(nextStr,target,cur+Long.valueOf(curStr),Long.valueOf(curStr),item+"+"+curStr,res);
                dfs(nextStr,target,cur-Long.valueOf(curStr),Long.valueOf(curStr),item+"+"+curStr,res);
                dfs(nextStr, target, cur-preCurDiff + preCurDiff*Long.valueOf(curStr),
                        preCurDiff*Long.valueOf(curStr), item + "*" + curStr, res);
            }
        }
    }
    /***
     * 困难--301、删除无效的括号
     */
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        // 统计多余的左括号和右括号数目
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '('){
                left++;
            }else if (s.charAt(i) == ')'){
                if (left>0) left--;
                else right++;
            }
        }
        dfs(s,0,left,right,"",res);
        return res;
    }
    public void dfs(String s, int index, int left, int right, String brackets, List<String> list){
        if (left<0 || right<0) return;
        if (index == s.length()){
            if (left !=0 || right!=0) return;
            if (isValid(brackets) && !list.contains(brackets)){
                list.add(brackets);
            }
            return;
        }
        char ch = s.charAt(index);
        if (ch == '('){
            dfs(s,index+1,left-1,right,brackets,list);//删除左括号
            dfs(s,index+1,left,right,brackets+ch,list);//删除左括号
        }else if (ch == ')'){
            dfs(s,index+1,left,right-1,brackets,list);
            dfs(s,index+1,left,right,brackets+ch,list);
        }else//其他字符，直接加上
            dfs(s,index+1,left,right,brackets+ch,list);
    }
    private boolean isValid(String s){
        int count=0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '('){
                count++;
            }else if (s.charAt(i) == ')'){
                count--;
                if (count<0) return false;
            }
        }
        return count == 0;
    }
    /***
     * 困难--329、矩阵中最长的递增路径
     */
    private int[][] dirs={{0,1},{1,0},{0,-1},{-1,0}};
    public int longestIncreasingPath(int[][] matrix) {
        if (matrix.length == 0) return 0;
        int m=matrix.length,n=matrix[0].length;
        int result=0;
        int[][] cache=new int[m][n];
        for (int i=0;i<m;i++){
            for (int j=0;j<n;j++){
                result=Math.max(result,dfs(matrix,i,j,cache));
            }
        }
        return result;
    }
    private int dfs(int[][] matrix, int i, int j,int[][] cache) {
        if (cache[i][j]!=0) return cache[i][j];
        for (int[] d : dirs) {
            int x=i+d[0],y=j+d[1];
            if (0<=x && x<matrix.length && 0<=y &&
            y<matrix[0].length && matrix[x][y] > matrix[i][j]){
                cache[i][j]=Math.max(cache[i][j],dfs(matrix,x,y,cache));
            }
        }
        return ++cache[i][j];
    }

    /***
     * 中等--351、安卓系统手势解锁
     */
    public int numberOfPatterns(int m,int n){
        int[][] skip=new int[10][10];
        skip[1][3]=skip[3][1]=2;
        skip[1][7]=skip[7][1]=4;
        skip[3][9]=skip[9][3]=6;
        skip[7][9]=skip[9][7]=8;
        skip[1][9]=skip[9][1]=skip[2][8]=skip[8][2]=skip[3][7]=skip[7][3]=skip[4][6]=skip[6][4]=5;
        boolean[] vis = new boolean[10];
        int res=0;
        for (int i=m;i<=n;i++){
            res+=DFS(vis,skip,1,i-1)*4;// 1, 3, 7, 9 are symmetric
            res+=DFS(vis,skip,2,i-1)*4;// 2, 4, 6, 8 are symmetric
            res+=DFS(vis,skip,5,i-1);// 5
        }
        return res;
    }
    public int DFS(boolean[] vis,int[][] skip,int cur,int remain){
        if (remain < 0) return 0;
        if (remain == 0) return 1;
        vis[cur]=true;
        int res=0;
        for (int i=1;i<=9;i++){
            // If vis[i] is not visited and (two numbers are adjacent or skip number is already visited)
            if (!vis[i] && (skip[cur][i] == 0 || vis[skip[cur][i]])){
                res+=DFS(vis,skip,i,remain-1);
            }
        }
        vis[cur]=false;
        return res;
    }

    /***
     *
     * @param args
     */
    public static void main(String[] args) {
        int[] arr={10,1,2,7,6,1,5};
        Dg dg = new Dg();
        List<List<Integer>> lists = dg.combinationSum3(3,9);
        System.out.println(lists.toString());
    }
}
class ListNode{
    int val;
    ListNode next;
    ListNode(int x){val =x;}
}

class TrieNode{
    private TrieNode[] links;
    private final int R=26;
    private boolean isEnd;

    public TrieNode(){
        links=new TrieNode[R];
    }

    public boolean containsKey(char ch){
        return links[ch - 'a']!=null;
    }

    public void put(char ch,TrieNode node){
        links[ch-'a']=node;
    }
    public TrieNode get(char ch){
        return links[ch - 'a'];
    }
    public void setEnd(){
        isEnd=true;
    }
    public boolean isEnd(){
        return isEnd;
    }
}
class Trie{
    private TrieNode root;//头节点
    public Trie(){//初始化头节点
        root=new TrieNode();
    }
    /**插入单词*/
    public void insert(String word){
        TrieNode node=root;//遍历树
        for(int i=0;i<word.length();i++){
            char curChar = word.charAt(i);
            if (!node.containsKey(curChar)){//没有包含就创建一个节点
                node.put(curChar,new TrieNode());
            }
            node=node.get(curChar);//向下遍历
        }
        node.setEnd();
    }

    /**查找一个前缀或者一个字符串是否在树中，返回查找节点的终止位置*/
    public TrieNode searchPrefix(String word){
        TrieNode node=root;
        for (int i=0;i<word.length();i++){
            char tmp = word.charAt(i);
            if (node.containsKey(tmp)){
                node=node.get(tmp);//向下遍历
            }else
                return null;
        }
        return node;
    }

    /**返回这个单词是否在trie中（必须要求是终结点）*/
    public boolean search(String word){
        TrieNode node = searchPrefix(word);
        return (node!=null) && (node.isEnd())?true:false ;
    }
    /** 返回是否有单词是以这个前缀开始的 */
    public boolean startsWith(String prefix) {
        TrieNode node = searchPrefix(prefix);
        return node != null?true:false;
    }
}
