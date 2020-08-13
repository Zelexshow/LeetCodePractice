package Array;

import netscape.javascript.JSUtil;
import sun.java2d.windows.GDIRenderer;

import java.util.*;

/***
 * 数组方面的问题
 */
public class Array {

    /***
     * 简单--26、删除排序数组中的重复项，要求原地返回，不能使用额外空间,返回新数组的长度
     */
    public int removeDuplicates(int[] nums){
        if (nums.length <=1) return nums.length;
        int len=nums.length,next=1,pre=0;
        while(next<len){
            if (nums[next] !=nums[pre]){
                nums[++pre]=nums[next++];//原地覆盖
            }else{
                next++;
            }
        }
        return pre+1;
    }
    /***
     * 中等--80、删除排序数组中的重复项-2，允许元素重复出现两次
     */
    public int removeDuplicates2(int[] nums) {
        int j=1,count=1;//j表示下一个要覆盖的索引
        for (int i=1;i<nums.length;i++){
            if (nums[i] == nums[i-1]){
                count++;
            }else{
                count=1;
            }
            if (count <= 2) nums[j++]=nums[i];//当个数小于等于2的时候就继续覆盖，
        }
        return j;
    }

    /***
     * 简单--217、存在重复元素
     */
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x:nums){
            if (set.contains(x)) return true;
            set.add(x);
        }
        return false;
    }
    /***
     * 简单--219、存在重复元素2
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        HashSet<Integer> set = new HashSet<>();
        for (int i=0;i<nums.length;i++){
            if (set.contains(nums[i])) return true;
            set.add(nums[i]);
            if (set.size()>k) set.remove(nums[i-k]);//窗口过期
        }
        return false;
    }
    /***
     * 中等--220、存在重复元素3
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeMap<Integer, Integer> maxMap = new TreeMap<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        TreeMap<Integer, Integer> minMap=new TreeMap<>();
        int len=nums.length;
        if (len == 0) return false;
        maxMap.put(nums[0],1);
        minMap.put(nums[0],1);
        for (int i=1;i<len;i++){
            //删除队头元素
            if (i>k){//容量为k+1
                remove(maxMap,nums[i-k-1]);
                remove(minMap,nums[i-k-1]);
            }
            if (maxMap.size() == 0) continue;
            long max= maxMap.firstKey();
            long min=minMap.firstKey();
            if (nums[i]>=max){
                if (nums[i]-max<=t) return true;
            }else if (nums[i]<=min){
                if (min-nums[i]<=t) return true;
            }else{
                for (int j=1;j<=k;j++){
                    if (i-j<0) break;
                    if (Math.abs((long) nums[i]-nums[i-j])<=t) return true;
                }
            }
            add(maxMap,nums[i]);
            add(minMap,nums[i]);
        }
       return false;
    }
    private void add(Map<Integer, Integer> treeMap, int num) {
        // TODO Auto-generated method stub
        Integer v = treeMap.get(num);
        if (v == null) {
            treeMap.put(num, 1);
        } else {
            treeMap.put(num, v + 1);
        }
    }

    private void remove(Map<Integer, Integer> treeMap, int num) {
        // TODO Auto-generated method stub
        Integer v = treeMap.get(num);
        if (v == 1) {
            treeMap.remove(num);
        } else {
            treeMap.put(num, v - 1);
        }
    }
    /***
     * 简单--27、移除元素
     */
    public int removeElement(int[] nums, int val) {
        int next=0;
        for (int j=0;j<nums.length;j++){
            if (nums[j]!=val){
                nums[next++]=nums[j];
            }
        }
        return next;
    }

    /***
     * 中等--31、下一个排列
     */
    public void nextPermutation(int[] nums){
        int index=0,nindex=0;
        int len=nums.length;
        boolean flag=false;
        for (int i=len-1;i>=1;i--){
            if (nums[i-1]<nums[i]) {
                index = i - 1;
                flag = true;
                break;
            }
        }
        if (!flag){//说明是降序排列，那么只有返回完全逆序
            int tmp=0;
            for (int i=0;i<len/2;i++){
                swap(nums,i,len-i-1);
            }
            return;
        }else{
            for (int i=len-1;i>=0;i--){
                if (nums[i]>nums[index]){//从右往左找到第一个大于index位置元素的索引
                    nindex=i;
                    swap(nums,index,nindex);//交换
                    break;
                }
            }
            //后面index位置后的所有元素全部逆序
            for (int i=index+1;i<(len-index-1)/2+index+1;i++){
                swap(nums,i,len-i+index);//交换从而逆序
            }
        }
    }

    private void swap(int[] nums,int left,int right){
        int temp=nums[left];
        nums[left]=nums[right];
        nums[right]=temp;
    }
    /***
     * 中等--33、搜索旋转排序数组,不含重复元素
     */
    public int search(int[] nums,int target){
        if (nums == null || nums.length == 0) return -1;
        int start=0;
        int end=nums.length-1;
        int mid;
        while(start <= end){
            mid=(start+end)/2+start;
            if (nums[mid] == target) return mid;
            //前半部分有序
            if (nums[mid]>=nums[start]){
                //target在前半部分
                if (target >= nums[start] && nums[mid]>=target) end=end-1;
                else start=mid+1;
            }else{
                if (target <= nums[end] && target >= nums[mid]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        return -1;
    }

    //二刷
    public int search1_2(int[] nums, int target) {

        if (nums == null || nums.length == 0) return -1;
        int left=0,right=nums.length-1;
        while(left<=right){//记住，括号内出结果的一定要写=号
            int mid=(right-left)/2+left;
            if (nums[mid] == target) return mid;
            if (nums[mid]>nums[right]){//说明位于前半部分
                if (nums[mid]>=target && target>=nums[left]){//说明要继续向右移动，注，最好写等号
                    right=mid-1;
                }else{
                    left=mid+1;
                }
            }else{//说明位于后半部分
                if (target>=nums[mid] && target<=nums[right]){//注，最好写等号
                    left=mid+1;
                }else{
                    right=mid-1;
                }
            }
        }
        return -1;
    }
    /***
     * 中等--81、搜索旋转排序数组2,含重复元素
     */
    public boolean search2(int[] nums, int target) {
        if (nums == null || nums.length == 0) return false;
        int start=0,end=nums.length-1;
        int mid;
        while(start <= end){
            mid=(start+end)/2+start;
            if (nums[mid] == target) return true;
            if (nums[start] == nums[mid]){
                start++;
                continue;
            }
            //前半部分有序
            if (nums[mid]>=nums[start]){
                //target在前半部分
                if (nums[mid]>=target && nums[start]<=target){
                    end=mid-1;
                }else{
                    start=mid+1;
                }
            }else{
                //target在后半部分
                if (nums[mid]<=target && target<=nums[end]){
                    start=mid+1;
                }else end=mid-1;
            }
        }
        return false;
    }
    /***
     * 中等--153、寻找旋转排序数组中的最小值
     */
    public int findMin(int[] nums) {//注意：找最小值是采用中间值和右边的值作比较
        int left=0;
        int right=nums.length-1;
        while(left<right){ /* 循环不变式，如果left == right，则循环结束 */
            int mid=left+(right-left)/2;
            if (nums[mid]>nums[right]){/* 中值 > 右值，最小值在右半边，收缩左边界 */
                left=mid+1;
            }else if (nums[mid]<nums[right]){/* 明确中值 < 右值，最小值在左半边，收缩右边界 */
                right=mid;/* 因为中值 < 右值，中值也可能是最小值，右边界只能取到mid处 */
            }
        }
        return nums[left];
        //发散思维，如果采用和左边的值作比较，就找最大值，然后最大值右边的就是最小值
    }

    /***
     * 困难--154、寻找旋转排序数组中的最小值2:包含重复值
     */
    public int findMin2(int[] nums) {
        int left=0,right=nums.length-1;
        while(left<right){
            int mid=left+(right-left)/2;
            if (nums[mid] > nums[right]){//说明位于前半段递增序列
                left=mid+1;
            }else if (nums[mid] < nums[right]) right=mid;//说明最小值位于左边
            else right=right-1;//增加去重操作
        }
        return nums[left];
    }
    /***
     * 中等--36、有效的数独
     * 小的三格宫索引公式：box_index = (row / 3) * 3 + columns / 3
     */
    public boolean isValidSudoku(char[][] board) {
        HashMap<Integer,Integer>[] rows=new HashMap[9];
        HashMap<Integer,Integer>[] cols=new HashMap[9];
        HashMap<Integer,Integer>[] cells=new HashMap[9];

        //初始化
        for (int i=0;i<9;i++){
            rows[i]=new HashMap<Integer, Integer>();
            cols[i]=new HashMap<Integer, Integer>();
            cells[i]=new HashMap<Integer, Integer>();
        }
        //开始遍历
        for (int i=0;i<9;i++){
            for (int j=0;j<9;j++){
                char num=board[i][j];
                if (num != ','){
                    int n=(int)num;
                    int cell_id=(i/3)*3+j/3;
                    rows[i].put(n,rows[i].getOrDefault(n,0)+1);
                    cols[j].put(n,cols[j].getOrDefault(n,0)+1);
                    cells[i].put(n,cells[cell_id].getOrDefault(n,0)+1);
                    if (rows[i].get(n)>1 || cols[j].get(n)>1
                            || cells[cell_id].get(n)>1) return false;//出现重复就报false
                }
            }
        }
        return true;

    }
    /***
     * 困难--41、缺失的第一个正整数
     */
    public int firstMissingPositive(int[] nums) {
        if (nums == null || nums.length == 0) return 1;
        Arrays.sort(nums);
        int i=0;
        while(i<nums.length && nums[i]<=0) i++;
        if (i == nums.length) return 1;
        if (nums[i]>1) return 1;//i此时为第一个大于0 的数的索引
        while(i+1<=nums.length-1){
            if (nums[i+1]-nums[i]<=1) i++;
            else{
                return nums[i]+1;
            }
        }
        return nums[i]+1;//说明是最后一个数的后一个数

    }

    //二刷缺失的第一个正数
    public int firstMissingPositive1_2(int[] nums){
        if (nums == null || nums.length == 0) return -1;
        Arrays.sort(nums);
        int ix=0;
        while(ix<nums.length && nums[ix]<=0) ix++;//跳过负数
        if (ix == nums.length || nums[ix] > 1) return 1;
        while(ix<=nums.length-2){
            if (nums[ix+1] - nums[ix] <= 1) ix++;
            else{
                return nums[ix]+1;
            }
        }
        return nums[ix]+1;//说明是最后一个数的后一个数


    }
    /***
     * 中等--48、旋转图像
     * 思路：由外向内一层一层的旋转
     */
    public void rotate(int[][] matrix) {
        int lr=0,lc=0,rr=matrix.length-1,rc=matrix[0].length-1;
        if (rr == 0 || matrix == null) return;;
        while(lc<rc){
            rotateEdge(matrix,lr++,lc++,rr--,rc--);
        }
    }
    public void rotateEdge(int[][] m,int tR,int tC,int dR,int dC){
        int times=dC-tC;
        int tmp=0;
        for (int i=0;i!=times;i++){
            tmp=m[tR][tC+i];//原始第一个点
            m[tR][tC+i]=m[dR-i][tC];//最后一个点到第一个点
            m[dR-i][tC]=m[dR][dC-i];
            m[dR][dC-i]=m[tR+i][dC];
            m[tR+i][dC]=tmp;
        }
    }
    /***
     * 中等--56、合并区间
     */
    public int[][] merge(int[][] intervals) {
        if (intervals.length<=1) return intervals;
        Arrays.sort(intervals,(arr1,arr2)-> arr1[0]-arr2[0]);
        int[] tmp=intervals[0];//创建临时数组
        List<int[]> res=new ArrayList<>();
        res.add(tmp);
        for (int[] ints : intervals) {
            int cur_begin=tmp[0];
            int cur_end=tmp[1];

            int next_begin=ints[0];
            int next_end=ints[1];

            //查看是否重叠
            if (cur_end >= next_begin){
                tmp[1]=Math.max(cur_end,next_end);
            }else{
                tmp=ints;
                res.add(tmp);
            }
        }
        return res.toArray(new int[res.size()][]);
    }
    //二刷
    public int[][] merge1_2(int[][] intervals){
        if (intervals.length <=1) return intervals;
        //首先排序,起点早的放前面
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0]-o2[0];
            }
        });
        ArrayList<int[]> res=new ArrayList<>();
        res.add(intervals[0]);
        int[] tmp=intervals[0];

        for (int[] arr:intervals){
            int cur_s=tmp[0];
            int cur_e=tmp[1];

            int next_s=arr[0];
            int next_e=arr[1];

            if (next_s>cur_e){//说明区间不重合
                res.add(arr);//直接加入
                tmp=arr;//同时更新当前的比较值
            }else{//说明区间重合
                tmp[1]=cur_e>=next_e?cur_e:next_e;//由于是数组，直接更改即可
            }
        }
        return res.toArray(new int[res.size()][]);

    }

    /***
     * 困难--57、插入区间
     */
    public int[][] insert(int[][] intervals,int[] newInterval){
        int newStart=newInterval[0],newEnd=newInterval[1];
        int ix=0,n=intervals.length;
        LinkedList<int[]> output=new LinkedList<>();//输出结果
        //先找到插入点
        while(ix<intervals.length && newStart>intervals[ix][0]) output.add(intervals[ix++]);

        //加入新的区间
        int[] interval=new int[2];
        if (output.isEmpty() || output.getLast()[1]<newStart){
            output.add(newInterval);
        }else{
            interval=output.removeLast();
            interval[1]=Math.max(interval[1],newEnd);
            output.add(interval);
        }
        while(ix<n){
            interval=intervals[ix++];
            int start=interval[0],end=interval[1];
            if (output.getLast()[1]<start) output.add(interval);
            else{
                interval=output.removeLast();
                interval[1]=Math.max(interval[1],end);
                output.add(interval);
            }
        }
        return output.toArray(new int[output.size()][]);
    }

    /***
     * 中等--54、螺旋矩阵
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> list = new ArrayList<>();
        if (matrix == null || matrix.length == 0) return list;
        int lr=0,lc=0,rr=matrix.length-1,rc=matrix[0].length-1;
        while(lr<=rr && lc<=rc){
            addEdge(matrix,lr++,lc++,rr--,rc--,list);
        }
        return list;
    }
    public void addEdge(int[][] m,int lr,int lc,int rr,int rc,List<Integer> list){
        if (lr == rr){//只有一行
            for (int i=lc;i<=rc;i++) list.add(m[lr][i]);
        }else if (lc ==rc){//只有一列
            for (int i=lr;i<=rr;i++) list.add(m[i][lc]);
        }else{
            int curR=lr,curC=lc;
            while(curC<rc){//打印上面的边，从左到右打印
                list.add(m[curR][curC++]);
            }
            while(curR<rr){//打印右边的边，从上到下打印
                list.add(m[curR++][curC]);
            }
            while(curC>lc){//打印下面的边，从右到左打印
                list.add(m[curR][curC--]);
            }
            while(curR>lr){//打印左边的边，从下到上打印
                list.add(m[curR--][curC]);
            }
        }
    }
    /***
     * 中等--59、螺旋矩阵2
     */
    public int[][] generateMatrix(int n) {
        int[][] res=new int[n][n];
        int start=1,lr=0,lc=0,rr=n-1,rc=rr;
        while(lr<=rr){
            start=addElements(res,start,lr++,lc++,rr--,rc--);//每次把上次留下来的值存住
        }
        return res;
    }
    private int addElements(int[][] res,int start,int lr,int lc,int rr,int rc){
        if(lr==rr){//注意！！！！
            res[lc][lr]=start;//边界条件，用于只含有一个数字的层数
        }else{
            int curR=lr,curC=lc;
            while(curC<rc){
                res[curR][curC++]=start++;
            }
            while(curR<rr){
                res[curR++][curC]=start++;
            }
            while(curC>lc){
                res[curR][curC--]=start++;
            }
            while(curR>lr){
                res[curR--][curC]=start++;
            }
        }
        return start;
    }
    /***
     * 中等--74、搜索二维矩阵
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix == null) return false;
        int rows=matrix.length,cols=matrix[0].length;
        int sr=0,sc=cols-1;
        while(sr<rows && sc>=0){
            if (matrix[sr][sc]>target){
                sc--;
            }else if (matrix[sr][sc]<target){
                sr++;
            }else
                return true;
        }
        return false;
    }

    /***
     * 中等--75、颜色分类
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) return;
        int left=-1;//确定数字为0 的右边界
        int right=nums.length;//确定数字为2的左边界
        int ix=0;
        while(ix<right){
            if (nums[ix] == 0) swap(nums,ix++,++left);
            if (nums[ix] == 2) swap(nums,ix,--right);
            else ix++;
        }
    }
    /***
     * 困难--149、直线上最多的点数
     */
    public int maxPoints(int[][] points){
        if (points.length < 3) return points.length;//只有两个点或者以下，直接返回
        //判断是否所有的点都相等
        int i=0;
        for (;i<points.length-1;i++){
            if (points[i][0] != points[i+1][0] || points[i][1] != points[i+1][1]) break;
        }
        if (i == points.length-1) return points.length;
        int max=0;
        for (i=0;i<points.length;i++){
            for (int j=i+1;j<points.length;j++){
                //i，j两点构成直线，开始验证其他点是否落在这条直线上
                //先排除两点不重合 的情况
                if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) continue;
                int tmpmax=0;
                for (int k=0;k<points.length;k++){
                    if (k!=i && k!=j){
                        if (test(points[i][0],points[i][1],points[j][0],points[j][1]
                                ,points[k][0],points[k][1])){
                            tmpmax++;
                        }
                    }
                }
                max=Math.max(max,tmpmax);
            }
        }
        return max;
    }
    //判断x,y是否在x1,x2。。。构成的直线上
    private boolean test(int x1, int y1, int x2, int y2, int x, int y) {
        return (long) (y2 - y1) * (x - x2) == (long) (y - y2) * (x2 - x1);
    }

    /***
     * 简单--189、右旋数组
     * 逻辑：整体翻转，翻转前部，翻转后部
     */
    public void rotate(int[] nums,int k){
        k%=nums.length;
        reverse(nums,0,nums.length-1);
        reverse(nums,0,k-1);
        reverse(nums,k,nums.length-1);
    }
    public void reverse(int[] arr,int l,int r){
        while(l<r){
            int tmp=arr[l];
            arr[l]=arr[r];
            arr[r]=tmp;
            l++;
            r--;
        }
    }
    /***
     * 中等--215、数组中第K个最大的元素
     */
    public int findKthLargest(int[] nums, int k) {
        //构建大顶堆
        PriorityQueue<Integer> heap = new PriorityQueue<>((i1, i2) -> i2 - i1);
        for (int num : nums) {
            heap.add(num);
            if (heap.size()>k) heap.poll();
        }
        int res=0;
        for (int i=0;i<k;i++){
            if (i == k-1) res = heap.poll();
            heap.poll();
        }
        return res;
    }

    //二刷
    public int findKthLargest1_2(int[] nums,int k){
        int len=nums.length;
        for (int parent=len/2-1;parent>=0;parent--){
            heapMax(nums,parent,len);//从下到上堆排序，构造大顶堆
        }
        int result=0;
        for (int i=0;i<k;i++){
            if (i == k-1){
                result=nums[0];
                break;
            }
            //把当前最大的值放在最后面
            int tmp=nums[len-i-1];
            nums[len-i-1]=nums[0];
            nums[0]=tmp;//这一波交换完后，第i+1大的最大值被放在了最后面 位置
            heapMax(nums,0,len-i-1);
        }
        return result;

    }
    //三刷
    public int findKthLargest1_3(int[] nums,int k){
        //第一次建立大顶堆
        int len=nums.length;
        for(int i=(len-1)/2;i>=0;i--){
            //从右到左，从下到上
            heapMax(nums,i,len);
        }//构建完成后，第一个元素就是最大的元素
        int result=0;

        for (int i=0;i<k;i++){
            if (i == k-1){
                result=nums[0];
                break;
            }
            //交换第一个元素和最后一个元素
            swap(nums,0,len-1-i);
            heapMax(nums,0,len-1-i);
        }

        return result;
    }
    //从上到下构造大顶堆
    public void heapMax(int[] nums,int start,int end){
        int val=nums[start];
        int leftChild=2*start+1;

        while(leftChild<end){//
            if (leftChild+1<end && nums[leftChild]<nums[leftChild+1]) leftChild=leftChild+1;
            if (val > nums[leftChild]) break;//大于就直接结束
            nums[start]=nums[leftChild];//把子节点大的移动上来
            start=leftChild;//继续向下寻找
            leftChild=2*start+1;
        }
        nums[start]=val;
    }
    /***
     * 中等--238、除自身以外数组的乘积
     */
    public int[] productExceptSelf(int[] nums) {
        int len=nums.length;
        if (nums == null || len ==0 ) return null;
        int[] up=new int[len];
        int[] down=new int[len];
        int[] res=new int[len];
        down[0]=1;
        for (int i=1;i<len;i++) down[i]=down[i-1]*nums[i-1];
        up[len-1]=1;
        for (int i=len-2;i>=0;i--) up[i]=up[i+1]*nums[i+1];
        for (int i=0;i<len;i++) res[i]=up[i]*down[i];
        return res;
    }
    /***
     * 中等--240、搜索二维数组
     */
    //搜索二维矩阵，诀窍，从右上角开始找
    public boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix == null) return false;
        int row=0,col=matrix[0].length-1;
        while(row<matrix.length && col>=0){
            if (matrix[row][col] > target){
                col--;
            }else if (matrix[row][col] < target) row++;
            else return true;
        }
        return false;
    }
    /***
     * 中等--274、H指数,未排序
     */
    public int hIndex(int[] citations) {
        Arrays.sort(citations);
        int ix=0;
        while(ix<citations.length && citations[citations.length-1-ix]>ix) ix++;
        return ix;
    }

    /***
     * 中等--275、H指数,已排序
     */
    public int hIndex2(int[] citations) {
        int ix=0;
        while(ix<citations.length && citations[citations.length-1-ix]>ix) ix++;
        return ix;
    }

    /***
     * 中等--280、摆动排序
     */
    public void wiggleSort(int[] nums){
        Arrays.sort(nums);
        for (int i=2;i<nums.length;i+=2){
            int tmp=nums[i];
            nums[i]=nums[i-1];
            nums[i-1]=tmp;
        }
    }
    /***
     * 中等--289、生命游戏
     */
    public void gameOfLife(int[][] board) {
        int[] neighbors={0,-1,1};

        int rows=board.length,cols=board[0].length;
        //创建复制的数组
        int[][] copyBoard=new int[rows][cols];

        for (int i=0;i<rows;i++){
            copyBoard[i]=Arrays.copyOfRange(board[i],0,cols-1);
        }

        for (int row=0;row<rows;row++){
            for (int col=0;col<cols;col++){
                //统计当前位置八个相邻位置的活细胞的数量
                int liveNeibors=0;

                for (int i=0;i<3;i++){
                    for (int j=0;j<3;j++){
                        if (!(neighbors[i] ==0 && neighbors[j] == 0)){
                            int r=(row+neighbors[i]);
                            int c=(col+neighbors[j]);

                            //查看相邻的细胞是否是活细胞
                            if ((r<rows && r>=0) && (c<cols && c>=0) && copyBoard[i][j] == 1)
                                liveNeibors++;
                        }
                    }
                }
                //规则1或者3
                if ((copyBoard[row][col] == 1) && (liveNeibors<2 || liveNeibors>3)){
                    board[row][col]=0;
                }
                //规则4
                if (copyBoard[row][col] == 0 && liveNeibors==3){
                    board[row][col]=1;
                }
            }
        }
    }

    /***
     * 困难--296、最佳碰头地点
     */
    public int minTotalDistance(int[][] grid) {
        List<Integer> xPoints = new ArrayList<>();
        List<Integer> yPoints = new ArrayList<>();

        //添加点的坐标
        for (int i=0;i<grid.length;i++){
            for (int j=0;j<grid[0].length;j++){
                if (grid[i][j] == 1){
                    xPoints.add(i);
                    yPoints.add(j);
                }
            }
        }
        return getMap(xPoints)+getMap(yPoints);
    }

    private int getMap(List<Integer> points){
        Collections.sort(points);
        int i=0,j=points.size()-1;
        int res=0;
        while(i<j){
            res+=points.get(j--)-points.get(i++);
        }
        return res;
    }
    /***
     * 简单--299、猜数字游戏
     */
    public String getHint(String secret, String guess) {
        StringBuilder sb = new StringBuilder();
        int a=0,b=0;
        int[] s=new int[10];
        int[] g=new int[10];
        for (int i=0;i<secret.length();i++){
            s[secret.charAt(i)-'0']++;
            g[secret.charAt(i)-'0']++;
            a+=secret.charAt(i) == guess.charAt(i)?1:0;
        }
        for (int i=0;i<s.length;i++){
            b+=Math.min(s[i],g[i]);//统计重合部分的个数为cows
        }
        return sb.append(a).append("A").append(b-a).append("B").toString();
    }
    /***
     * 中等--311、稀疏矩阵的乘法
     */
    public int[][] multiply(int[][] A, int[][] B) {
        int[][] C=new int[A.length][B.length];
        for (int i=0;i<A.length;i++){
            for (int j=0;j<A[i].length;j++){
                if (A[i][j] == 0) continue;
                for (int k=0;k<B[0].length;k++){
                    if (B[j][k] == 0) continue;
                    C[i][k]+=A[i][j]*B[j][k];
                }
            }
        }
        return C;
    }
    /***
     * 中等--324、摆动排序2,把之前的等于去掉了
     */
    public void wiggleSort2(int[] nums) {
        Arrays.sort(nums);//先排序
        int len=nums.length,i=0;
        //左边的数组大于等于右边数组的长度
        int[] smaller=new int[len%2==0?len/2:len/2+1],bigger=new int[len/2];
        //复制
        System.arraycopy(nums,0,smaller,0,smaller.length);
        System.arraycopy(nums,smaller.length,bigger,0,len/2);

        //穿插
        for (;i<len/2;i++){
            nums[2*i]=smaller[smaller.length-1-i];
            nums[2*i+1]=bigger[len/2-1-i];
        }
        if (len%2!=0) nums[2*i]=smaller[smaller.length-1-i];
    }

    /***
     * 中等--347、前K个高频元素
     */
    public int[] topKFrequent(int[] nums,int k){
        HashMap<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num,count.getOrDefault(num,0)+1);
        }
        PriorityQueue<Integer> heap = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return count.get(o1) - count.get(o2);
            }
        });
        for (Integer key : count.keySet()) {
            heap.add(key);
            if (heap.size()>k) heap.poll();
        }
        //输出
        List<Integer> res=new LinkedList<>();
        while(!heap.isEmpty()){
            res.add(heap.poll());
        }
        Collections.reverse(res);
        int[] r=new int[k];
        for (int i=0;i<k;i++) r[i]=res.get(i);
        return r;
    }

    /***
     * 简单--350、两个数组的交集2
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int n : nums1) {
            map.put(n,map.getOrDefault(n,0)+1);
        }
        int ix=0;
        for (int n : nums2) {
            if (map.getOrDefault(n,0)>0){
                nums1[ix++]=n;
                int tmp=map.get(n);
                map.put(n,--tmp);
            }
        }
        return Arrays.copyOfRange(nums1,0,ix);
    }
    /***
     * 困难--354、俄罗斯套娃信封问题
     */
    public int maxEnvelopes(int[][] envelopes){
        //按照w进行排序
        Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) return o2[1]-o1[1];
                else return o1[0]-o2[0];
            }
        });
        //按照h进行LIS操作
        int[] secondDim=new int[envelopes.length];
        for (int i=0;i<envelopes.length;i++) secondDim[i]=envelopes[i][1];
        return lengthOfLIS(secondDim);
    }

    /***
     * 中等--300、最长上升子序列
     */
    public int lengthOfLIS(int[] nums){
        if (nums == null || nums.length == 0) return 0;
        //dp[i]表示以nums[i]为结尾的最长上升子序列的长度
        int[] dp=new int[nums.length];
        Arrays.fill(dp,1);//将所有的子序列长度都设为初值1，即只包含自己本身
        int res=1;
        for (int i=1;i<nums.length;i++){
            for (int j=0;i<i;j++){
                if (nums[j]<nums[i]){
                    dp[i]=Math.max(dp[i],dp[j]+1);
                }
                res=Math.max(dp[i],res);
            }
        }
        return res;
    }

    //二刷
    public int lengthOfLIS1_2(int[] nums){
        if (nums == null || nums.length == 0) return 0;
        int[] dp=new int[nums.length];
        Arrays.fill(dp,1);
        int max=1;
        for (int i=1;i<dp.length;i++){
            for (int j=0;j<i;j++){
                if (nums[j]<nums[i]){
                    dp[i]=Math.max(dp[i],dp[j]+1);
                }
            }
            max=Math.max(max,dp[i]);
        }
        return max;
    }
    /***
     * 中等--361、炸弹人
     */
    public int maxKilledEnemies(char[][] grid){
        if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;
        int row=0;
        int[] col=new int[grid[0].length];
        int max=0;
        for (int i=0;i<grid.length;i++){
            for (int j=0;j<grid[0].length;j++){
                if (grid[i][j] == 'W') continue;
                if (j == 0 || grid[i][j-1] == 'W') row=calRowEnemy(grid,i,j);
                if (i == 0 || grid[i-1][j] == 'W') col[j]=calColEneymy(grid,i,j);
                if (grid[i][j] == '0') max=Math.max(max,row+col[j]);
            }
        }
        return max;
    }
    public int calRowEnemy(char[][] grid,int i,int j){
        int res=0;
        while(j<grid[0].length && grid[i][j] != 'W'){
            res+=res+(grid[i][j] == 'E'?1:0);
            j++;
        }
        return res;
    }
    public int calColEneymy(char[][] grid,int i,int j){
        int res=0;
        while (i<grid.length && grid[i][j]!='W') {
            res = res + (grid[i][j]=='E'? 1 : 0);
            i++;
        }
        return res;
    }

    /***
     * 剑指offer--和为S的连续正数序列
     */
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum){
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        int left=1,right=2;
        while(left<right){
            int curSum=(left+right)*(right-left+1)/2;
            if (curSum == sum){
                ArrayList<Integer> rec = new ArrayList<>();
                for (int i=left;i<=right;i++){
                    rec.add(i);
                }
                result.add(rec);
                left++;
            }else if (curSum<sum) right++;//右指针往后滑动
            else left++;
        }
        return result;
    }

    /**
     * 剑指offer -- 和为sum的两个数字
     * */
    public ArrayList<Integer> FindNumbsersWithSum(int[] array,int sum){
        ArrayList<Integer> list = new ArrayList<>();
        if (array == null || array.length < 2) return list;
        int i=0,j=array.length-1;

        while(i<j){
            if (array[i]+array[j] == sum){
                list.add(array[i]);
                list.add(array[j]);
                return list;
            }else if (array[i]+array[j] > sum) j--;
            else i++;
        }
        return list;
    }

    /****
     * 简单--88、合并两个有序数组
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int len1=nums1.length;
        int[] tmp_nm1=new int[m];
        System.arraycopy(nums1,0,tmp_nm1,0,m);
        int p1=0,p2=0;
        int ix=0;
        while(p1<m && p2<n){
            nums1[ix++]=nums1[p1]<nums2[p2]?nums1[p1++]:nums2[p2++];
        }
        if (p1<m){
            System.arraycopy(tmp_nm1,p1,nums1,p1+p2,m+n-p1-p2);
        }
        if (p2<n) System.arraycopy(nums2,p2,nums1,p1+p2,m+n-p1-p2);
    }
    /*public int findMin(int[] nums) {
        int left=0,right=nums.length-1;
        while(left<right){
            int mid=(right-left)/2+left;
            if (nums[mid]>nums[right]){
                left=mid+1;
            }else if (nums[mid] < nums[right]){
                right=mid;
            }
        }
        return nums[left];
    }*/

    /****
     * 简单--1299、将每个元素替换成右侧最大的元素
     */
    public int[] replaceElements(int[] arr) {
        int len=arr.length;
        int[] res=new int[len];
        res[len-1]=-1;//最后一个元素默认为-1
        int rigthMax=arr[len-1];//记录右侧最大值
        for (int i=len-2;i>=0;i--){
            res[i]=rigthMax;
            rigthMax=arr[i]>rigthMax?arr[i]:rigthMax;//更新右侧最大值
        }
        return res;
    }

    /****
     * 中等--560、和为K的子数组
     */
    public int subarraySum(int[] nums, int k) {
        int[] sum=new int[nums.length+1];
        sum[0]=0;//sum[i];表示从0-i-1号索引的和
        for (int i=1;i<=nums.length;i++){
            sum[i]=sum[i-1]+nums[i-1];
        }
        int count=0;
        for (int left=0;left<nums.length;left++){
            for (int right=left;right<nums.length;right++){
                if (sum[right+1]-sum[left] == k){
                    count++;
                }
            }
        }
        return count;
    }

    /***
     *简单--283、移动零
     */
    public void moveZeroes(int[] nums) {
        if (nums == null) return;
        int j=0;
        for (int i=0;i<nums.length;i++){
            if (nums[i]!=0){
                int tmp=nums[i];
                nums[i]=nums[j];
                nums[j++]=tmp;
            }
        }
    }

    /***
     * 困难--128、最长连续序列
     */
    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        //加入到set中便于去重
        for (int i=0;i<nums.length;i++) set.add(nums[i]);
        int cnt=0,result=0,tmp1=0,tmp2=0;
        //以当前元素为起点分别向前 后遍历，寻找连续的数字
        for (int j=0;j<nums.length;j++){
            if (!set.contains(nums[j])) continue;
            cnt=0;
            tmp1=nums[j];
            tmp2=nums[j]+1;
            while(set.contains(tmp1)){
                cnt++;
                set.remove(tmp1);
                tmp1--;
            }
            while(set.contains(tmp2)){
                cnt++;
                set.remove(tmp2);
                tmp2++;
            }
            result=cnt>result?cnt:result;
        }
        return result;
    }

    /***
     *
     */
    List<List<Integer>> result=null;
    public List<List<Integer>> printList(int n){
         result= new ArrayList<>();
        if (n == 0) return result;
        if (n == 1){
            List<Integer> cur = new ArrayList<>();
            cur.add(1);
            result.add(cur);
            return result;
        }
        helper(n,new ArrayList<Integer>());
        return result;
    }
    public void helper(int target,List<Integer> cur){
        if (target == 0){
            result.add(new ArrayList<>(cur));
            return;
        }
        if (target < 0) return;

        if (target>=2){
            cur.add(2);//加入到结果中
            helper(target-2,cur);
            cur.remove(cur.size()-1);//回溯
        }
        if (target >= 1){
            cur.add(1);
            helper(target-1,cur);
            cur.remove(cur.size()-1);
        }

    }

    /***
     * 中等--55、跳跃游戏
     */
    public boolean canJump(int[] A) {
        //贪心策略为：从右到左遍历数组，找到最左的能够到达终点的位置的元素，最后判断是否能够到达
        //0索引位置
        //lastPos标记为最左的能否到达终点的索引（0）
        int lastPos=A.length-1;
        for (int i=A.length-1;i>=0;i--){
            if (i+A[i] >= lastPos) lastPos=i;
        }
        return lastPos == 0;
    }

    /***
     * 困难--45、跳跃游戏2
     */
    public int jump(int[] nums) {
        int end=0,steps=0,maxPosition=0;
        for (int i=0;i<nums.length-1;i++){
            //记录当前位置可以跳到的最远位置，注意是最远位置的下标
            maxPosition=Math.max(maxPosition,i+nums[i]);
            //如果遍历到的位置下标到达我们上面标记的最大位置，我们已经跳跃到了最大位置，此时需要更新我们的最大位置，并且步数需要加1
            if (i == end){
                end=maxPosition;
                steps++;
            }
        }
        return steps;
    }

    /****
     * 困难--42、接雨水
     */
    public int trap(int[] height){
        int[] max_left=new int[height.length];
        int[] max_right=new int[height.length];
        int sum=0;
        for (int i=1;i<height.length;i++){
            max_left[i]=max_left[i-1]>height[i-1]?max_left[i-1]:height[i-1];
        }
        for (int i=height.length-2;i>=0;i--){
            max_right[i]=max_right[i+1]>height[i+1]?max_right[i+1]:height[i+1];
        }
        for (int i=1;i<height.length-1;i++){
            int less=Math.min(max_left[i],max_right[i]);
            if (less>height[i]) sum+=less-height[i];
        }
        return sum;
    }

    /***
     * 简单--496、下一个更大的元素
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Stack<Integer> stack = new Stack<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i=0;i<nums2.length;i++){
            while (!stack.isEmpty() && nums2[i] > stack.peek()){//维护递增栈（栈顶到栈底）
                map.put(stack.pop(),nums2[i]);
            }
            stack.push(nums2[i]);
        }
        while (!stack.isEmpty()){
            map.put(stack.pop(),-1);
        }
        int[] res=new int[nums1.length];
        for (int i=0;i<nums1.length;i++){
            res[i]=map.get(nums1[i]);
        }
        return res;
    }

    /***
     * 中等--503、下一个更大的元素2
     */
    public int[] nextGreaterElements(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int[] res=new int[nums.length];
        Arrays.fill(res,-1);
        for (int i=0;i<2*nums.length;i++){
            int num=nums[i%nums.length];
            while(!stack.isEmpty() && num>nums[stack.peek()]){
                res[stack.pop()]=num;
            }
            if (i<nums.length){
                stack.push(i);
            }
        }
        return res;
    }

    /***
     * 剑指offer--数组中的逆序对
     * 利用归并排序的思想
     */
    int count;//用于统计
    public int reversePairs(int[] nums) {
        count=0;
        if (nums != null){
            divPairs(nums,0,nums.length-1);
        }
        return count;
    }
    public void divPairs(int[] arr,int start,int end){
        if (start >= end) return;
        int mid=(start+end)>>1;
        divPairs(arr,start,mid);
        divPairs(arr,mid+1,end);
        mergePairs(arr,start,mid,end);
    }
    public void mergePairs(int[] arr,int start,int mid,int end){
        int i=start,j=mid+1,k=0;
        int[] tmp=new int[end-start+1];//用来存放新的结果
        while(i<=mid && j<= end){
            if (arr[i]<=arr[j]){
                tmp[k++]=arr[i++];
            }else{
                tmp[k++]=arr[j++];
                count+=mid-i+1;//因为到目前为止，已经将原数组排好序了
            }
        }
        while(i <= mid) tmp[k++]=arr[i++];
        while(j <= end) tmp[k++]=arr[j++];
        //将原数组变成有序
        System.arraycopy(tmp,0,arr,start,end-start+1);
    }

    /***
     * 中等--986、区间列表的交集
     */
    public int[][] intervalIntersection(int[][] A,int[][] B){
        List<int[]> ans=new ArrayList<>();
        int i=0,j=0;
        while(i<A.length && j<B.length){
            int low=Math.max(A[i][0],B[j][0]);
            int high=Math.min(A[i][1],B[j][1]);
            if (low <= high){
                ans.add(new int[]{low,high});
            }
            if (A[i][1]<B[j][1])
                i++;
            else j++;
        }
        return ans.toArray(new int[ans.size()][]);
    }
    public static void main(String[] args) {
        Array ins = new Array();
        ArrayList<ArrayList<Integer>> result = ins.FindContinuousSequence(100);

        int[] arr={1,2,3,5,4,6,6,8,7};
        ins.nextPermutation(arr);
        System.out.println(Arrays.toString(arr));

        List<List<Integer>> lists = ins.printList(2);
        System.out.println(lists);

        int[] a1={4,1,2},a2={1,3,4,2};
        int[] res = ins.nextGreaterElement(a1, a2);
        System.out.println(Arrays.toString(res));
        Set<Integer> set = new HashSet<>();
    }
}
