package sort;

import Array.Array;

import java.util.Arrays;
import java.util.Random;

public class Sort {
    //基本建立在大顶堆上
    public void heapSort(int[] arr){
        if (arr == null || arr.length<=1) return;
        int len=arr.length;
        for (int i=(len-1)/2;i>=0;i--){
            //从第一个非叶子节点从下至上，从右至左调整结构
            heapify(arr,i,len);
        }
        for(int i=len-1;i>0;i--){
            int tmp=arr[0];
            arr[0]=arr[i];
            arr[i]=tmp;
            //重新对堆进行调整
            heapify(arr,0,i);
        }
    }

    public void heapify(int[] arr,int parent,int len){//len表示的是堆队尾的元素
        int tmp=arr[parent];//保存本节点
        int lchild=parent*2+1;
        while(lchild<len){
            int rchild=lchild+1;
            if (rchild<len && arr[lchild]<arr[rchild]){
                lchild++;
            }
            if(tmp>arr[lchild]) break;
            arr[parent]=arr[lchild];//将子节点的值赋给父节点

            parent=lchild;
            lchild=lchild*2+1;//向下迭代
        }
        arr[parent]=tmp;//将父节点放到末尾
    }
    /***
     * 计数排序
     */
    public int[] countingSort(int[] array){
        //定义数组的最大值
        int max = array[0];
        for(int i=1;i<array.length;i++){
            if(array[i]>max){
                max=array[i];
            }
        }
        //根据数组最大值确定统计数组的长度
        int[] countArray = new int[max+1];
        //遍历数组，把统计数组填上
        for(int i1=0;i1<array.length;i1++){
            countArray[array[i1]]++;
        }
        //遍历统计数组输出结果
        int index=0;
        int[] sortArray = new int[array.length];
        for(int j=0;j<countArray.length;j++){
            for(int v=0;v<countArray[j];v++){
                sortArray[index++]=j;
            }
        }
        return sortArray;
    }

    /***
     * 基于partition的快排
     */

    public void quickSortBasePartition(int[] arr,int l,int r){
        if (arr == null || arr.length < 1) return;
        int len=arr.length;//数组的长度
        Random randUtils = new Random();
        //首先随机选择一个索引作为标准值传递到首位
        if (l<r){
            swap(arr,randUtils.nextInt(r-l+1)+l,l);//随机选择索引作为标准值
            int[] limit=partition(arr,l,r);
            quickSortBasePartition(arr,l,limit[0]-1);
            quickSortBasePartition(arr,limit[1]+1,r);
        }
    }
    public int[] partition(int[] arr,int start,int right){
        int standard=arr[start];
        int low=start-1,high=right+1;
        int ix=start;
        while(ix<high){
            if (arr[ix]<standard){
                swap(arr,++low,ix++);
            }else if (arr[ix]>standard){
                swap(arr,--high,ix);
            }else ix++;
        }
        return new int[]{low+1,high-1};//这两个边界是等于区的边界
    }
    public void swap(int[] arr,int l,int r){
        int tmp=arr[l];
        arr[l]=arr[r];
        arr[r]=tmp;
    }

    /***
     * 归并排序
     */
    public void mergeSort(int[] arr,int l,int r){
        if (l<r){
            int mid=(r-l)/2+l;
            mergeSort(arr,l,mid);
            mergeSort(arr,mid+1,r);
            merge(arr,l,mid,r);
        }
    }
    public void merge(int[] arr,int left,int mid,int right){
        int[] tmp=new int[right-left+1];
        int i=0,j=left,k=mid+1;
        while(j <= mid && k<=right){
            if (arr[j]>arr[k]){
                tmp[i++]=arr[k++];
            }else tmp[i++]=arr[j++];
        }
        while(j<=mid){
            tmp[i++]=arr[j++];
        }
        while(k<=right){
            tmp[i++]=arr[k++];
        }
        for (int t=0;t<i;t++){
            arr[left+t]=tmp[t];
        }
    }
    public static void main(String[] args) {
        Sort ins = new Sort();
        int[] array= new int[]{4,4,6,9,2,5,0,10,9,4,6,9,1,0};

        ins.quickSortBasePartition(array,0,array.length-1);
        System.out.println(Arrays.toString(array));
        /*int[] sortArray=ins.countingSort(array);*/
//        System.out.println(Arrays.toString(sortArray));



    }


}
