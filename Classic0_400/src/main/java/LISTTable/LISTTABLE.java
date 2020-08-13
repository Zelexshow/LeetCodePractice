package LISTTable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/***
 * 链表系列
 */
public class LISTTABLE {

    public static class ListNode{
        int val;
      ListNode next;
      public ListNode(int x) { val = x; }
    }

    public static class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }
    /**
     * 简单--83. 删除排序链表中的重复元素
     */
    public ListNode deleteDuplicates(ListNode head){
        ListNode cur=head,post;//post指向下一个节点
        while(cur!=null){
            while(cur.next !=null && cur.next.val == cur.val){
                post=cur.next.next;
                cur.next=post;
            }
            cur=cur.next;
        }
        return head;
    }
    //二刷
    public ListNode deleteDuplicates1_2(ListNode head){
        if (head == null || head.next == null) return head;
        ListNode cur=head;
        while(cur!=null){
            while(cur.next !=null  && cur.val == cur.next.val){
                cur.next=cur.next.next;
            }
            cur=cur.next;
        }
        return head;
    }

    /**
     * 中等--84、删除排序链表中的重复元素
     */
    public ListNode deleteDuplicates2(ListNode head){
        if (head == null || head.next == null) return head;
        ListNode newHead=new ListNode(-1);//作为新的头节点，防止首节点就出现重复
        newHead.next=head;
        ListNode cur=head;
        ListNode pre=newHead;
        while(cur!=null && cur.next !=null){
            if (cur.next.val !=cur.val){
                pre=cur;
            }else{
                while(cur.next != null && cur.next.val == cur.val){
                    cur=cur.next;
                }
                pre.next=cur.next;
            }
            cur=cur.next;
        }
        return newHead.next;
    }
    /**
     * 简单--88、合并链表
     */
    public ListNode mergeTwoLists(ListNode l1,ListNode l2){
        //版本1：迭代法
        /*ListNode nH = new ListNode(0);
        ListNode cur=nH;
        while (l1!=null && l2!=null){
            if (l1.val <= l2.val){
                cur.next=l1;
                l1=l1.next;
            }else{
                cur.next=l2;
                l2=l2.next;
            }
            cur=cur.next;
        }
        if (l1!=null) cur.next=l1;
        if (l2!=null) cur.next=l2;
        return nH.next;*/
        //版本2：递归法
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val<l2.val){
            l1.next=mergeTwoLists(l1.next,l2);
            return l1;
        }else{
            l2.next=mergeTwoLists(l1,l2.next);
            return l2;
        }
    }

    /***
     * 困难--23、合并K个有序链表
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        return mergelists(lists,0,lists.length-1);
    }
    /**采用归并思想*/
    public ListNode mergelists(ListNode[] lists,int left,int right){
        if (left == right) return lists[left];
        int mid=(right-left)/2+left;
        ListNode l = mergelists(lists,left,mid);
        ListNode r = mergelists(lists, mid + 1, right);
        ListNode res = merge(l, r);
        return res;

    }
    //合并逻辑
    public ListNode merge(ListNode l1,ListNode l2){
        ListNode nH = new ListNode(0);
        ListNode cur=nH;
        while (l1!=null && l2!=null){
            if (l1.val <= l2.val){
                cur.next=l1;
                l1=l1.next;
            }else{
                cur.next=l2;
                l2=l2.next;
            }
            cur=cur.next;
        }
        if (l1!=null) cur.next=l1;
        if (l2!=null) cur.next=l2;
        return nH.next;
    }

    /****
     * 简单--141、环形链表
     */
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode fast=head,slow=head;
        while(fast != null && fast.next != null){
            fast=fast.next.next;
            slow=slow.next;
            if (slow == fast) return true;
        }
        return false;
    }
    /***
     * 中等--142.环形链表2
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return null;
        ListNode meetNode = meetingNode(head);
        if (meetNode == null) return null;
        ListNode slow=meetNode;
        ListNode fast=head;
        while (fast != slow){
            fast=fast.next;
            slow=slow.next;
        }
        return fast;//再次相交，第一个入环点就是
    }
    //寻找相遇节点，如果无环，返回null
    public ListNode meetingNode(ListNode head) {
        ListNode slow=head;
        ListNode fast=head;
        while(fast!=null && fast.next!=null){
            slow=slow.next;
            fast=fast.next.next;
            if (slow == fast) return slow;
        }
        return null;//说明没环
    }

    /***
     * 简单--206、翻转链表
     */
    public ListNode reverseList(ListNode head) {
        /**迭代版本*/
        /*if (head == null || head.next == null) return head;
        ListNode cur=head,pre=null;//一个指向前驱节点，一个指向当前节点
        while(cur != null){
            ListNode nex=cur.next;
            cur.next=pre;
            pre=cur;
            cur=nex;
        }
        return pre;*/
        /**递归版本*/
        if (head == null || head.next == null) return head;
        ListNode nH = reverseList(head.next);//反转后的头节点
        ListNode cur=nH;
        while (cur.next != null){
            cur=cur.next;
        }
        cur.next=head;
        head.next=null;//记得置空
        return nH;
    }

    /***
     * 中等--92、反转链表（给定区间）
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) return null;
        ListNode cur=head,pre=null;

        while (m>1){
            pre=cur;
            cur=cur.next;
            m--;
            n--;
        }
        ListNode con=pre,tail=cur;
        ListNode third=null;

        while (n>0){
            third=cur.next;
            cur.next=pre;
            pre=cur;
            cur=third;
            n--;
        }
        if (con != null){
            con.next=pre;
        }else {
            head=pre;
        }
        tail.next=cur;
        return head;

    }

    /***二刷*/
    public ListNode reverseBetween2(ListNode head,int m,int n){
        if (head == null) return head;
        //获得链表长度
        ListNode cur=head,pre=null;
        //找到起始点
        while(m>1){
            pre=cur;
            cur=cur.next;
            m--;
            n--;
        }
        ListNode con=pre,tail=cur;
        ListNode third=null;
        while(n>0){//开始翻转后面的链表
            third=cur.next;//记录下一个节点
            cur.next=pre;
            pre=cur;
            cur=third;
            n--;
        }
        if (con != null){
            con.next=pre;
        }else{
            head=pre;
        }
        tail.next=cur;//将反转后的节点指向后半部分的节点
        return head;
    }

    /***
     * 中等--318、奇偶链表
     */
    public ListNode oddEvenList(ListNode head) {
        if (head == null) return null;
        ListNode odd=head,even=head.next,evenHead=even;
        while(even!=null && even.next != null){
            odd.next=even.next;
            odd=odd.next;
            even.next=odd.next;
            even=even.next;
        }
        odd.next=evenHead;
        return head;
    }

    /***
     * 困难--25、K个一组翻转链表
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy=new ListNode(-1);
        dummy.next=head;
        ListNode pre=dummy;
        ListNode end=dummy;
        while (end.next!=null){
            for (int i=0;end!=null&&i<k;i++) end=end.next;
            if (end == null) break;
            ListNode start=pre.next;
            ListNode next=end.next;//保存后一个节点
            end.next=null;//置空，准备翻转
            pre.next=reverse(start);
            start.next=next;//连接到下一段
            pre=start;
            end=pre;
        }
        return dummy.next;
    }
    private ListNode reverse(ListNode head){
        ListNode pre=null,cur=head,nex=null;
        while(cur!=null){
            nex=cur.next;
            cur.next=pre;
            pre=cur;
            cur=nex;
        }
        return pre;
    }
    /***二刷*/
    public ListNode reversrGroup1_2(ListNode head,int k){
        ListNode dummy = new ListNode(-1);
        dummy.next=head;
        ListNode pre=dummy;
        ListNode end=dummy;
        while(end.next != null){
            for (int i=0;end != null && i<k;i++) end = end.next;//移动到末尾
            if (end == null) break;//不足k个直接翻转
            ListNode start=pre.next;
            ListNode nex=end.next;//保存后一个节点
            end.next = null;//置空，准备翻转
            pre.next = reverse(start);
            start.next=nex;//连接到下一个节点
            pre=start;
            end=pre;
        }
        return dummy.next;
    }
    /***
     * 中等--143、重排链表
     */
    public void reorderList(ListNode head){
        //1、首先找到链表的中点
        ListNode middle = findMiddle(head);
        ListNode left=head;
        ListNode right=middle.next;
        middle.next=null;
        //2、将中点后半段的链表翻转
        right=reverse(right);

        //3、重新合并两个链表
        merge2(left,right);
    }
    public ListNode findMiddle(ListNode head){
        ListNode slow=head,fast=head;
        while(fast !=null && fast.next !=null){
            fast=fast.next.next;
            slow=slow.next;
        }
        return slow;
    }
    /**合并链表*/
    public void merge2(ListNode l1,ListNode l2){
        ListNode lcur,rcur;
        while(l1.next!=null && l2 !=null){
            //保存节点
            lcur=l1.next;
            rcur=l2.next;
            //移动节点
            l1.next=l2;
            l2.next=lcur;
            l1=lcur;
            l2=rcur;
        }
    }

    /**
     * 简单--160、相交链表
     * */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        int n=0;//用来表示链表的长度差
        ListNode curA=headA,curB=headB;
        while(curA.next!=null){
            curA=curA.next;
            n++;
        }
        while(curB.next!=null){
            curB =curB.next;
            n--;
        }
        if (curA != curB) return null;//说明没有相交的点
        curA= n>0?headA:headB;//定位谁是长的链表
        curB= curA == headA?headB:headA;//两个链表都回到链表头部

        n=Math.abs(n);
        while(n!=0){
            n--;
            curA=curA.next;
        }
        while (curA != curB){
            curA = curA.next;
            curB=curB.next;
        }
        return curA;
    }
    //二刷
    public ListNode getIntersectionNode1_2(ListNode h1,ListNode h2){
        if (h1 == null || h2 == null) return null;
        int n=0;
        ListNode cur1=h1,cur2=h2;
        while(cur1.next != null){
            n++;
            cur1=cur1.next;
        }
        while(cur2.next != null){
            n--;
            cur2=cur2.next;
        }
        if (cur1 != cur2) return null;//两个链表最后一个节点不相等，说明不相交
        cur1=n>0?h1:h2;//cur1代表长的
        cur2=cur1==h1?h2:h1;//cur2代表短的
        n=Math.abs(n);
        while(n>0){
            cur1=cur1.next;
            n--;
        }
        while (cur1 != cur2){
            cur1=cur1.next;
            cur2=cur2.next;
        }
        return cur1;
    }
    /**
     * 中等--2、两数相加
     * */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        ListNode dummy=new ListNode(-1);
        ListNode head=dummy;
        int resn=0;
        while(l1 != null && l2 != null){
            int sum=l1.val+l2.val;
            ListNode tmp=new ListNode((sum+resn)%10);
            resn=(sum+resn)/10;
            head.next=tmp;
            head=head.next;
            l1=l1.next;
            l2=l2.next;
        }
        while (l1 != null){
            head.next=new ListNode((resn+l1.val)%10);
            resn=(resn+l1.val)/10;
            head=head.next;
            l1=l1.next;
        }
        while(l2 != null){
            head.next=new ListNode((resn+l2.val)%10);
            resn=(resn+l2.val)/10;
            head=head.next;
            l2=l2.next;
        }
        if (resn == 1){
            head.next=new ListNode(resn);
        }
        return dummy.next;
    }
    //两数相加
    public ListNode addTwoNumbers1_2(ListNode l1,ListNode l2){
        ListNode tmp = new ListNode(-1);
        ListNode head=tmp;
        int count=0;
        while(l1 != null && l2 != null){
            count=(l1.val+l2.val+count)/10;
            int curSum=(l1.val+l2.val+count)%10;
            head.next=new ListNode(curSum);
            l1=l1.next;
            l2=l2.next;
        }
        while(l1 != null){
            head.next=new ListNode((l1.val+count)%10);
            head=head.next;
            count=(l1.val+count)/10;
            l1=l1.next;
        }
        while (l2 != null){
            head.next=new ListNode((l2.val+count)%10);
            head=head.next;
            count=(l2.val+count)/10;
            l2=l2.next;
        }
        if (count == 1){
            head.next=new ListNode(1);
            head=head.next;
        }
        return tmp.next;
    }

    /***
     * 简单--234、回文链表
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        ListNode slow=head,fast=head;
        while(fast.next !=null && fast.next.next != null){
            slow=slow.next;
            fast=fast.next.next;
        }
        ListNode mid=slow;//标记中间节点
        ListNode mid_cur=mid.next;//标记后半部分的头节点
        mid.next=null;//置空当前节点

        ListNode last_pre=null;//后半部分节点的前一个节点
        ListNode last_nex=null;//当前遍历节点的下一个节点

        while(mid_cur !=null){
            last_nex=mid_cur.next;//记录下一个节点
            mid_cur.next=last_pre;//指向前驱节点
            last_pre=mid_cur;
            mid_cur=last_nex;
        }
        //此时last_pre就是后半边链表的头节点
        //再从头节点开始遍历，对比两个链表
        ListNode nHead=last_pre;
        ListNode preHead=head;
        boolean result=true;//标记是否事回文链表
        while(nHead != null && preHead != null){
            if (nHead.val != preHead.val){
                result=false;
                break;
            }
            nHead=nHead.next;
            preHead=preHead.next;
        }
        return result;
    }

    /***
     * 中等--138、复制带随机指针的链表
     */
    public Node copyRandomList(Node head){
        if (head == null) return head;
        Map<Node, Node> map = new HashMap<>();
        Node copyHead = new Node(head.val);
        map.put(head,copyHead);
        Node cur = head;
        Node copyCur=copyHead;
        while(cur.next != null){
            copyCur.next=new Node(cur.next.val);//复制下一个节点
            cur=cur.next;
            copyCur=copyCur.next;
            map.put(cur,copyCur);
        }
        //压入随机节点
        cur=head;
        copyCur=copyHead;
        while(cur != null){
            copyCur.random=map.get(cur.random);
            cur=cur.next;
            copyCur=copyCur.next;
        }
        return copyHead;
    }

    public static void main(String[] args) {
        LISTTABLE ins = new LISTTABLE();
        ListNode head = new ListNode(0);
        ListNode cur=head;
        for (int i=1;i<=5;i++){
            cur.next= new ListNode(i);
            cur=cur.next;
        }
        cur=head;
        while(cur!=null){
            System.out.println(cur.val+",");
            cur=cur.next;
        }
        System.out.println("-------------");
        cur=ins.reverseList(head);
        while(cur!=null){
            System.out.println(cur.val+",");
            cur=cur.next;
        }
    }
}
