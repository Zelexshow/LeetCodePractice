package bt;

import Array.Array;

import java.util.*;

/***
 * 二叉树
 */
public class BTree {

    /**
     * 困难--297. 二叉树的序列化与反序列化
     */
    public String  serialize(TreeNode root){
        if (root == null) return "null,";
        String res=String.valueOf(root.val)+",";
        res+=serialize(root.left);
        res+=serialize(root.right);
        return res;
    }
    /**反序化二叉树*/
    public TreeNode deserialize(String data){
        String[] values = data.split(",");
        Queue<String> queue = new LinkedList<>();
        for (int i=0;i<values.length;i++) queue.offer(values[i]);
        return reconPreOrder(queue);
    }
    public TreeNode reconPreOrder(Queue<String> queue){
        String s=queue.poll();
        if ("null".equals(s)) return null;
        TreeNode head = new TreeNode(Integer.valueOf(s));
        head.left=reconPreOrder(queue);
        head.right=reconPreOrder(queue);
        return head;
    }
    /**
     * 简单--225、翻转二叉树
     * 自下而上
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left=right;
        root.right=left;
        return root;
    }
    /**
     * 中等--230、二叉搜索树中第K小的元素
     * 利用中序遍历，找到第k个元素
     */
    public int kthSmallest(TreeNode root, int k) {
        List<Integer> list = new ArrayList<>();
        helper(root,list);
        if (list.size()<k){ return  -1;}
        return list.get(k-1);
    }
    public void helper(TreeNode root,List<Integer> inorder){
        if (root == null) return;
        helper(root.left,inorder);
        inorder.add(root.val);
        helper(root.right,inorder);
    }
    /**
     * 简单--257、二叉树的所有路径
     */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root == null) return res;
        helper(res,root,"");
        return res;

    }
    public void helper(List<String> res,TreeNode root,String cur){
        if (root.left == null && root.right == null) res.add(new String(cur+root.val));
        cur+=root.val+"->";
        if (root.left != null) helper(res,root.left,cur);
        if (root.right !=null) helper(res,root.right,cur);
    }

    /**
     * 简单--270、最接近的二叉搜索树值
     * 二分策略
     */
    public int closestValue(TreeNode root, double target) {
        int val,closet=root.val;
        while(root != null){
            val=root.val;
            closet=Math.abs(val-target)<Math.abs(closet-target)?val:closet;
            root=target<closet?root.left:root.right;
        }
        return closet;
    }
    /***
     * 中等--285二叉搜索树的后继节点
     */
    public TreeNode inorderSuccessor(TreeNode root,TreeNode p){
        if (root == null) return null;
        if (p.val >= root.val) return inorderSuccessor(root.right,p);
        else{
            TreeNode left = inorderSuccessor(root.left, p);
            return left == null?root:left;
        }
    }
    /***
     * 中等--298、二叉树的最长连续序列
     */
    public int longestConsecutive(TreeNode root){
        if (root == null) return 0;
        int[] res=new int[]{0};
        TreeNode pre=null;
        find(root,res,1,pre);
        return res[0];
    }
    private void find(TreeNode root,int[] res,int count,TreeNode pre){
        if (root == null) return;
        if (pre !=null){
            if (root.val == pre.val+1) count++;
            else count=1;
        }
        pre=root;
        res[0]=Math.max(res[0],count);
        find(root.left,res,count,pre);
        find(root.right,res,count,pre);
    }

    /***
     * 中等--314、二叉树的垂直遍历
     */
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;

        HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();

        LinkedList<TreeNode> queue = new LinkedList<>();
        LinkedList<Integer> level = new LinkedList<>();//level相当于列

        queue.offer(root);
        level.offer(0);
        int minlevel=0;
        int maxlevel=0;
        while(!queue.isEmpty()){
            TreeNode p = queue.poll();
            int l = level.poll();

            minlevel = Math.min(minlevel, 1);
            maxlevel = Math.min(maxlevel, 1);

            if (map.containsKey(l)){
                map.get(l).add(p.val);
            }else{
                ArrayList<Integer> list = new ArrayList<>();
                list.add(p.val);
                map.put(l,list);
            }

            if (p.left != null){//添加子节点及对应的列数
                queue.offer(p.left);
                level.offer(l-1);
            }
            if (p.right!=null){
                queue.offer(p.right);
                level.offer(l+1);
            }
        }
        for (int i=minlevel;i<=maxlevel;i++){
            if (map.containsKey(i)){
                result.add(map.get(i));
            }
        }
        return result;
    }

    /***
     * 困难--124、二叉树的最大路径和
     */
    int sum=Integer.MIN_VALUE;//作为遍历的记录变量
    public int maxPathSum(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return root.val;
        getMaxSum(root);
        return sum;
    }

    /****
     * 最大路径的三种情况：
     * 1、左子树的路径+当前节点的值
     * 2、右子树的路径+当前节点的值
     * 3、左子树的路径+右子树的路径+当前节点的值
     * 4、只有当前节点的路径
     */
    public int getMaxSum(TreeNode root){
        if (root == null) return 0;
        int left = Math.max(getMaxSum(root.left), 0);//计算左子树的最大路径值
        int right = Math.max(getMaxSum(root.right), 0);//右子树的值
        sum=Math.max(sum,left+right+root.val);//更新最大值是否有变化

        return Math.max(left,right)+root.val;//只能返回左右子树中路径较大的值
    }

    //二刷
    public int getMaxSum1_2(TreeNode root){
        if (root == null) return 0;
        int left=Math.max(getMaxSum1_2(root.left),0);
        int right=Math.max(getMaxSum1_2(root.right),0);
        sum= Math.max(sum,root.val+left+right);
        return Math.max(left,right)+root.val;
    }

    /***
     * 二叉树的右视图
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if (root == null) return list;
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            int size=queue.size();
            for (int i=0;i<size;i++){
                TreeNode tmp=queue.poll();
                if (i == size-1) list.add(tmp.val);//每次取当前层最后一个数加入到结果中
                if (tmp.left!=null) queue.offer(tmp.left);
                if (tmp.right!=null) queue.offer(tmp.right);
            }
        }
        return list;
    }


    /***
     * 简单--108、将有序数组转化成二叉搜索树
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null) return null;
        return getSortBTree(nums,0,nums.length-1);
    }
    public TreeNode getSortBTree(int[] nums,int start,int end){
        if (start>end) return null;
        int mid=(end-start)/2+start;
        TreeNode root=new TreeNode(nums[mid]);
        root.left=getSortBTree(nums,start,mid-1);
        root.right=getSortBTree(nums,mid+1,end);
        return root;
    }
    /***
     * 中等--236、二叉树的最近公共祖先
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //还是有几种可能：
        //1、位于root的左子树
        //2、位于root的右子树
        //3、就是root本身
        if (root == null || root == q || root == p) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left == null) return right;
        if (right == null) return left;

        //到了这种情况，室友可能是p,q分别位于root的左右子树当中
        //不在左子树，也不在右子树，那么只有可能是root
        return root;
    }


    /****
     * 中等--515、在每个树行中找最大值
     */
    public List<Integer> largestValues(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> values = new ArrayList<>();
        if (root != null)
            queue.add(root);//入队
        while (!queue.isEmpty()) {
            int max = Integer.MIN_VALUE;
            int levelSize = queue.size();//每一层的数量
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();//出队
                max = Math.max(max, node.val);//记录每层的最大值
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            values.add(max);
        }
        return values;
    }

    /***
     * 简单--剑指offer 55、二叉树的深度
     */
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }

    /***
     * 简单--111、二叉树的最小深度
     */
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        //1.左孩子和有孩子都为空的情况，说明到达了叶子节点，直接返回1即可
        if (root.left == null && root.right == null) return 1;
        int l=minDepth(root.left);
        int r=minDepth(root.right);
        //2.如果左孩子和由孩子其中一个为空，那么需要返回比较大的那个孩子的深度
        if (root.left == null || root.right == null) return l+r+1;
        //3.最后一种情况，也就是左右孩子都不为空，返回最小深度+1即可
        return Math.min(l,r)+1;
    }

    /***
     * 中等--958、二叉树的完全性验证
     */
    public boolean isCompleteTree(TreeNode root) {
        if (root == null) return true;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean left=false;
        while(!queue.isEmpty()){
            TreeNode tmp = queue.poll();
            if (tmp.left == null && tmp.right != null) return false;
            if (left && (tmp.left != null || tmp.right != null)) return false;
            if (tmp.left !=null) queue.offer(tmp.left);
            if (tmp.right !=null) queue.offer(tmp.right);
            else left=true;
        }
        return true;
    }
    /***
     * 简单--101、对称二叉树
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) return true;
        return helper(root.left,root.right);
    }
    public boolean helper(TreeNode left,TreeNode right){
        if (left == null && right == null) return true;
        if (left == null) return false;
        if (right == null) return false;
        return left.val == right.val && helper(left.left,right.right) && helper(left.right,right.left);
    }

    /***
     * 剑指offer34--二叉树中和为某一值的路径
     */
    LinkedList<Integer> path=new LinkedList<>();
    List<List<Integer>> res=new LinkedList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        recur(root,sum);
        return res;
    }
    public void recur(TreeNode root,int tar){
        if (root == null) return;
        path.add(root.val);
        tar-=root.val;
        if (tar == 0 && root.left == null && root.right == null)
            res.add(new LinkedList<Integer>(path));
        recur(root.left,tar);
        recur(root.right,tar);
        path.removeLast();
    }
    //二刷

    /***
     * 中等--103、二叉树的锯齿形层次遍历
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return ans;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        //记录层数的奇偶性
        int cnt=0;
        while (!queue.isEmpty()){
            List<Integer> tmp=new ArrayList<>();
            int len=queue.size();
            for (int i=0;i<len;i++){
                TreeNode node = queue.poll();
                if (cnt % 2 == 0) tmp.add(node.val);
                else tmp.add(0,node.val);//奇数层就一直放在前面
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            cnt++;
            ans.add(tmp);
        }
        return ans;
    }
    //二刷
    public List<List<Integer>> zigzagLevelOrder1_2(TreeNode root){
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();//队列
        queue.offer(root);
        int level=1;
        while(!queue.isEmpty()){
            int size=queue.size();//确定size;
            LinkedList<Integer> tmp = new LinkedList<>();
            while(size>0){
                TreeNode cur = queue.poll();
                size--;
                if (level % 2 == 0){//偶数层
                    tmp.add(0,cur.val);
                }else{//奇数层
                    tmp.add(cur.val);
                }
                if (cur.left != null) queue.add(cur.left);
                if (cur.right != null) queue.add(cur.right);
            }
            result.add(tmp);
            level++;//层数加1
        }
        return result;
    }
    /***
     * 简单--100、相同的树
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;//如果都没节点了就为true
        if (p == null || q == null || p.val != q.val) return false;//返回false
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
    }

}
class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
  }
