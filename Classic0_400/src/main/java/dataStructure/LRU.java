package dataStructure;

import java.util.HashMap;
import java.util.Map;

public class LRU {
    class LRUNode{
        private int key;
        private int val;
        private LRUNode pre;
        private LRUNode next;

        LRUNode(int key,int val){
            this.key=key;
            this.val=val;
        }
    }
    private int capacity;
    private int count;
    private LRUNode dummy;//头节点
    private LRUNode tail;//尾节点
    private Map<Integer,LRUNode> map;

    //移除头节点
    private void removeHead(){
        map.remove(dummy.next.key);//首先移除map中的元素
        dummy.next=dummy.next.next;//然后移除链表中的元素
        if (dummy.next!=null) dummy.next.pre=dummy;
    }
    //添加尾节点
    private void appendTail(LRUNode node){
        tail.next=node;
        node.pre=tail;
        tail=node;
    }
    //获取值
    private int get(int key){
        if (!map.containsKey(key)) return -1;
        LRUNode node = map.get(key);
        if (node != tail){
            LRUNode pre = node.pre;
            pre.next=node.next;
            node.next=pre;
            appendTail(node);
        }
        return node.val;
    }
    //存放值
    private void put(int key,int val){
        LRUNode newNode = new LRUNode(key, val);
        if (map.containsKey(key)){
            LRUNode oldNode = map.get(key);
            if (oldNode != tail){//先判断是否在队尾
                LRUNode pre = oldNode.pre;
                pre.next=oldNode.next;
                oldNode.next.pre=pre;
            }else tail=tail.pre;//跳到上一个
        }else{
            if (count<capacity) count++;
            else removeHead();//移除头节点
        }
        appendTail(newNode);
        map.put(key,newNode);
    }
    public LRU(int capacity){
        this.capacity=capacity;
        this.count=0;
        dummy=new LRUNode(0,0);
        tail=dummy;
        map=new HashMap<>();
    }
}
