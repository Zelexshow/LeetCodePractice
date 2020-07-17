package BFS;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

/***
 * BFS相关问题
 */
public class BFS {
    /****
     * 中等--127、单词接龙
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return 0;
        HashSet<String> visited = new HashSet<>();
        LinkedList<String> queue = new LinkedList<>();

        queue.offer(beginWord);
        visited.add(beginWord);

        int count=0;//用来统计个数
        while(queue.size()>0){
            int size=queue.size();
            ++count;
            for (int i=0;i<size;i++){
                String start = queue.poll();
                for (String s:wordList){
                    //已经遍历过的就不再遍历
                    if (visited.contains(s)){
                        continue;
                    }
                    //不能转换的就跳过
                    if (!canConvert(start,s)){
                        continue;
                    }
                    //可以转换，并且能转换成endWord,则返回count;
                    if (s.equals(endWord)){
                        return count+1;
                    }
                    //保存访问过的单词，同时把单词放进队列，用于下一层的访问
                    visited.add(s);
                    queue.offer(s);
                }
            }
        }
        return 0;
    }
    public boolean canConvert(String start,String s){
        if (start.length() != s.length()) return false;
        int count=0;
        for (int i=0;i<s.length();i++){
            if (start.charAt(i) != s.charAt(i)){
                count++;
                if (count>1) return false;
            }
        }
        return count==1;
    }

}
