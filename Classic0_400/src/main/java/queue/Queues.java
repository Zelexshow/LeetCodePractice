package queue;

import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;

public class Queues {

    /***
     * 困难--239、滑动窗口的最大值
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        LinkedList<Integer> queue = new LinkedList<>();
        if (nums.length * k == 0) return new int[0];
        int[] res=new int[nums.length-k+1];//最终返回结果
        int index=0;
        for (int i=0;i<nums.length;i++){
            if (!queue.isEmpty() && i-k+1>queue.peek()){
                queue.pollFirst();//移除队头元素
            }
            while(queue.size() > 0 && nums[queue.peekLast()] < nums[i]){
                queue.pollLast();//移除所有的元素
            }
            queue.offerLast(i);//将元素加到队尾
            if (i >= k-1) res[index++]=nums[queue.peek()];//填充数据，等到窗口形成
        }
        return res;
    }
    //二刷
    public int[] maxSlid2(int[] nums,int k){
        if (nums.length*k == 0) return new int[0];
        int[] result=new int[nums.length-k+1];
        LinkedList<Integer> queue=new LinkedList<>();
        int ix=0,count=0;
        while(ix<nums.length){
            if (!queue.isEmpty() && ix-queue.peek() >= k) queue.pollFirst();//队首
            while(nums[ix]>nums[queue.peekLast()] && !queue.isEmpty()){
                queue.pollLast();
            }
            queue.addLast(ix);
            if (ix>= k-1) result[count++]=nums[queue.peek()];//队首始终放置最大的元素
        }
        return result;
    }
    /***
     * 困难--295、数据流的中位数
     */

    int size,winSum=0,count=0;

}
class MedianFinder{

    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;

    public MedianFinder(){
        maxHeap=new PriorityQueue<>((a,b)-> b-a);//大顶堆
        minHeap=new PriorityQueue<>((a,b)-> a-b);//小顶堆
    }
    public void addNum(int num){
        if (maxHeap.size() == 0 || num <=  maxHeap.peek()){
            maxHeap.offer(num);
        }else{
            minHeap.offer(num);
        }
        //大顶堆的尺寸必须大于小顶堆
        if (maxHeap.size() > minHeap.size()+1){
            minHeap.add(maxHeap.poll());
        }else if (maxHeap.size() < minHeap.size()){
            maxHeap.add(minHeap.poll());
        }
    }
    public double findMedian(){
        if (maxHeap.size() != minHeap.size()){
            return maxHeap.peek();
        }else {
            return maxHeap.peek()/2.0+minHeap.peek()/2.0;
        }
    }
}
