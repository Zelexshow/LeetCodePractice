package BinaryFind;

/***
 * 围绕二分法进行
 */
public class BinaryFind {
    /***
     * 中等--162、寻找峰值，
     */
    //方法1：线性查找，只需要比较n[i]是否大于n[i+1]就行
    /*public int findPeakElement(int[] nums) {
        for (int i=0;i<nums.length-1;i++){
            if (nums[i]>nums[i+1]) return i;
        }
        return nums.length-1;
    }*/
    //方法2、二分法
    //根据左右指针计算中间位置 m，并比较 m 与 m+1 的值，如果 m 较大，
    // 则左侧存在峰值，r = m，如果 m + 1 较大，则右侧存在峰值，l = m + 1
    public int findPeakElement(int[] nums) {
       int l=0,r=nums.length-1;
       while(l<r){
           int mid=l+(r-l)/2;
           if (nums[mid]>nums[mid+1]) r=mid;
           else l=mid+1;
       }
       return l;
    }
    /***
     * 中等--第一个错误的版本--简单
     */
    public int firstBadVersion(int n){
        int l=1,r=n;
        while(l<r){
            int mid=l+(r-l)/2;
            if (!isBadVersion(mid)){
                l=mid+1;
            }else {
                r=mid;
            }
        }
        return l;
    }
    boolean isBadVersion(int mid){
        return false;
    }
    /***
     * 中等--287、寻找重复数
     * 根据抽屉原则，cnt[i]代表数组中小于等于i的个数，如果cnt[i]<=i,那么不存在，如果cnt[i]>i,则必存在
     */
    public int findDuplicate(int[] nums){
        int low=0,high=nums.length-1;
        while(low<high){
            int mid=(high-low)/2+low;
            int count=0;
            for (int i=0;i<nums.length;i++)
                if (nums[i]<=mid) count++;
            if (count>mid){
                high=mid;
            }else
                low=mid+1;
        }
        return low;
    }
    /***
     * 困难--4、两排序数组的中位数
     * 相当于求两数组中第K小个数，k=(len1+len2)/2
    */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int left = (n + m + 1) / 2;
        int right = (n + m + 2) / 2;

        //将偶数和奇数的情况合并，如果是奇数，会求两次同样的 k 。
        return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;

    }
    private int getKth(int[] nums1,int start1,int end1,int[] nums2,int start2,int end2,int k){
        int len1=end1-start1+1;
        int len2=end2-start2+1;

        //让len1的长度一定小于len2,这样就能保证如果数组空了，一定是len1
        if (len1>len2) return getKth(nums2,start2,end2,nums1,start1,end1,k);
        if (len1 == 0) return nums2[start2+k-1];
        if (k == 1) return Math.min(nums1[start1],nums2[start2]);

        int i=start1+Math.min(len1,k/2)-1;
        int j=start2+Math.min(len2,k/2)-1;

        if (nums1[i]>nums2[j]){
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        }else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }

    /***
     * 中等--34. 在排序数组中查找元素的范围
     */
    public int[] searchRange(int[] nums, int target) {
        int[] targetR={-1,-1};
        int l=0,r=nums.length-1,mid=0;
        boolean find=false;
        while(l<=r){
            mid=(r-l)/2+1;
            if (nums[mid] == target){
                find=true;
                break;
            }else if (nums[mid]>target){
                r=mid-1;
            }else {
                l=mid+1;
            }
        }
        if (find){
            l=r=mid;
            while (l>=0 && nums[l]==nums[mid]) l--;
            while(r<nums.length && nums[r] == nums[mid]) r++;
            targetR[0]=l+1;
            targetR[1]=r-1;
        }
        return targetR;
    }
    /***
     * 困难--302、包含全部黑色素的最小矩阵
     */
    public int minArea(char[][] image,int x,int y){
        if(image == null || image.length == 0 || image[0].length == 0) return 0;
        int m=image.length,n=image[0].length;
        if (x<0 || x>=m || y<0 || y>=n) return 0;
        int up=binarySearch(image,0,x,0,n,true,true);
        int down=binarySearch(image,x+1,m,0,n,true,false);
        int left=binarySearch(image,0,y,up,down,false,true);
        int right=binarySearch(image,y+1,x,up,down,false,false);
        return (right-left)*(down-up);
    }
    private int binarySearch(char[][] image,int low,int high,int min,int max,boolean searchHor,boolean searchFirstOne){
        while(low<high){//searchHor表示水平列搜索
            int k=min,mid=(high-low)/2+low;
            while(k<max && (searchHor?image[mid][k]:image[k][mid]) == '0') ++k;
            if (k<max == searchFirstOne) high=mid;//searchFirstOne表示搜索的是第一个
            else low=mid+1;
        }
        return low;
    }

    /***
     * 中等--378、有序矩阵中第K小的元素
     */
    public int kthSmallest(int[][] matrix, int k) {
        int row=matrix.length,col=matrix[0].length;
        int left=matrix[0][0],right=matrix[row-1][col-1];
        while(left<right){
            int mid=(left+right)/2;
            //找二维矩阵中小于等于mid的元素的总数
            int count = findNotBiggerThanMid(matrix, mid, row, col);
            if (count<k){
                //第k小的数在右半部分，且不包括mid
                left=mid+1;
            }else {
                //第k小的数在左半部分
                right=mid;
            }
        }
        return right;
    }
    private int findNotBiggerThanMid(int[][] matrix, int mid, int row, int col) {
        //以列为单位，找到每一列最后一个<=mid的数即可知道每一列有多少个数<=mid
        int i=row-1;
        int j=0;
        int count=0;
        while(i>=0 && j<col){
            if (matrix[i][j] <= mid){
                //第j列有i+1个元素<=mid
                count+=i+1;
                j++;
            }else{
                //第j列目前的数大于mid,需要继续在当前列上找
                i--;
            }
        }
        return count;
    }
    public static void main(String[] args) {
        int[] arr={2, 4, 5, 2, 3, 1, 6, 7};
        BinaryFind bf = new BinaryFind();
        char[][] image={{'0','0','1','0'},{'0','1','1','0'},{'0','1','0','0'}};
        int area = bf.minArea(image, 0, 2);
        System.out.println(area);
        int duplicate = bf.findDuplicate(arr);
        System.out.println(duplicate);
    }
}
