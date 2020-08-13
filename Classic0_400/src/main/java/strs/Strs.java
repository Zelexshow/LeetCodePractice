package strs;


import com.sun.deploy.security.SelectableSecurityManager;
import com.sun.xml.internal.fastinfoset.util.ValueArrayResourceException;

import java.util.*;

/***
 * 字符串相关问题
 */
public class Strs {
    /***
     * 简单--字符串相加
     */
    public String addStr(String num1,String num2){
        StringBuilder sb = new StringBuilder();
        int i=num1.length()-1,j=num2.length()-1,carry=0;
        while(i >= 0 || j>= 0){
            int n1=i>=0?num1.charAt(i)-'0':0;
            int n2=j>=0?num2.charAt(j)-'0':0;
            int tmp=n1+n2+carry;
            carry=tmp/10;
            sb.append(tmp%10);
            i--;
            j--;
        }
        if (carry == 1) sb.append(1);
        return sb.reverse().toString();
    }

    /***
     * 中等--151、翻转字符串中的单词
     */
    public String reverseWords(String s){
        char[] chars = s.toCharArray();
        int len = chars.length;
        if (s == null|| len == 0) return s;
        reverse(chars,0,len-1);
        //先整体翻转
        for (int i=0;i<len;i++){
            int low=i;
            while(i<len && chars[i] != ' '){
                i++;
            }
            reverse(chars,low,i-1);
        }
        return new String(chars);

    }
    public void reverse(char[] arr,int l,int r){
        while(l<r){
            char tmp=arr[l];
            arr[l]=arr[r];
            arr[r]=tmp;
            l++;
            r--;
        }
    }

    /***
     * 简单--205、同构字符串
     */
    public boolean isIsomorphic(String s,String t){
        return isIsomorphicHelper(s,t) && isIsomorphicHelper(t,s);
    }
    private boolean isIsomorphicHelper(String s, String t){
        Map<Character, Character> map = new HashMap<>();
        int n=s.length();
        for (int i=0;i<n;i++){
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            if (map.containsKey(c1)){
                if (map.get(c1)!=c2){
                    return false;
                }
            }else{
                map.put(c1,c2);
            }
        }
        return true;
    }

    /***
     * 简单--243、最短单词距离
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        int ix1=-1,ix2=-1,dis=Integer.MAX_VALUE;
        for (int i=0;i<words.length;i++){
            if (words[i].equals(word1)){
                ix1=i;
                if (ix2 !=-1) dis=Math.min(dis,ix1-ix2);
            }
            if (words[i].equals(word2)){
                ix2=i;
                if (ix1 !=-1) dis=Math.min(dis,ix2-ix1);
            }
        }
        return dis;
    }

        /***
         * 中等--244、最短单词距离2
         */
        public class WordDistance{
            Map<String,List<Integer>> map=new HashMap<>();
            public WordDistance(String[] words){
                //统计每个单词出现的下标存入到hash表中
                for (int i=0;i<words.length;i++){
                    List<Integer> cnt = map.get(words[i]);
                    if (cnt == null){
                        cnt=new ArrayList<>();
                    }
                    cnt.add(i);
                    map.put(words[i],cnt);
                }
            }
            public int shortest(String word1,String word2){
                List<Integer> l1 = map.get(word1);
                List<Integer> l2 = map.get(word2);

                int dis=Integer.MAX_VALUE;
                int i=0,j=0;
                while(i<l1.size() && j<l2.size()){
                    dis=Math.min(Math.abs(l1.get(i)-l2.get(j)),dis);
                    if (l1.get(i)<l2.get(j)) i++;
                    else j++;
                }
                return dis;
            }
        }

    /***
     * 中等--249、移位字符串分组
     */
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<List<String>>();
        HashMap<String, ArrayList<String>> map
                = new HashMap<String, ArrayList<String>>();

        for(String s: strings){
            char[] arr = s.toCharArray();
            if(arr.length>0){
                int diff = arr[0]-'a';
                for(int i=0; i<arr.length; i++){
                    if(arr[i]-diff<'a'){
                        arr[i] = (char) (arr[i]-diff+26);
                    }else{
                        arr[i] = (char) (arr[i]-diff);
                    }
                }
            }

            String ns = new String(arr);
            if(map.containsKey(ns)){
                map.get(ns).add(s);
            }else{
                ArrayList<String> al = new ArrayList<String>();
                al.add(s);
                map.put(ns, al);
            }
        }

        for(Map.Entry<String, ArrayList<String>> entry: map.entrySet()){
            Collections.sort(entry.getValue());
        }

        result.addAll(map.values());
        return result;
    }

    /***
     * 中等--251、展开二维向量
     */
    public class Vector2D {
        List<Iterator<Integer>> its;
        int cur=0;
        public Vector2D(List<List<Integer>> vec2d){
            this.its=new ArrayList<>();
            for (List<Integer> l : vec2d) {
                if (l.size()>0){//只将非空的加入到数组中
                    this.its.add(l.iterator());
                }
            }
        }
        public int next(){
            Integer res = its.get(cur).next();
            if (!its.get(cur).hasNext()){
                cur++;
            }
            return res;
        }
        public boolean hasNext(){
            return cur<its.size() && its.get(cur).hasNext();
        }
    }

    /***
     * 中等--271、字符串的编解码
     */
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for (String s:strs){
            sb.append(intToString(s));
            sb.append(s);
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        int i=0,n=s.length();
        ArrayList<String> output = new ArrayList<>();
        while(i<n){
            int length=stringToInt(s.substring(i,i+4));
            i+=4;
            output.add(s.substring(i,i+length));
            i+=length;
        }
        return output;
    }

    private String intToString(String s) {
        int x=s.length();
        char[] bytes = new char[4];//用四个字节存储字符串长度的二进制码
        for (int i=3;i>-1;--i){
            bytes[3-i]=(char)(x>>(i*8) & 0xff);
        }
        return new String(bytes);
    }
    private int stringToInt(String i){
        int result=0;
        for (char b:i.toCharArray()){
            result = (result << 8)+(int)b;
        }
        return result;
    }

    /***
     * 中等--288、单词的唯一缩写
     */
    public class ValidWordAbbr{
        Map<String,Set<String>> map;
        public ValidWordAbbr(String[] dictionary){
            map=new HashMap<>();
            for (String s : dictionary) {
                String abbr = getAbbr(s);
                if (!map.containsKey(abbr)){
                    map.put(abbr,new HashSet<>());
                }
                map.get(abbr).add(s);
            }
        }
        public boolean isUnique(String word){
            String abbr = getAbbr(word);
            if (!map.containsKey(abbr) || (map.get(abbr).contains(word) &&
            map.get(abbr).size() == 1)){
                return true;
            }else{
                return false;
            }
        }
        public String getAbbr(String s){
            if (s.length()<3) return s;
            int len=s.length();
            return s.substring(0,1)+(len-2)+s.substring(len-1);
        }
    }

    /***
     * 简单--290、单词规律
     */
    public boolean wordPattern(String pattern, String str) {
        if (pattern.length() == 0 || str.length() == 0) return false;
        String[] arr1 = pattern.split("");
        String[] arr2 = str.split(" ");
        if (arr1.length != arr2.length) return false;
        return helper(arr1,arr2) && helper(arr2,arr1);
    }
    public boolean helper(String[] arr1,String[] arr2){
        Map<String, String> map = new HashMap<>();
        for (int i=0;i<arr1.length;i++){
            String key = arr1[i];
            if (map.containsKey(key)){
                if (!map.get(key).equals(arr2[i])){
                    return false;
                }
            }else{
                map.put(key,arr2[i]);
            }
        }
        return true;
    }

    /***
     * 中等--293、翻转游戏
     */
    public List<String> generatePossibleNextMoves(String s) {
        List<String> res=new ArrayList<>();
        for (int i=1;i<s.length();i++){
            if (s.charAt(i) == '+' && s.charAt(i-1) == '+')
                res.add(s.substring(0,i-1)+"--"+s.substring(i+1));
        }
        return res;
    }

    /***
     * 中等--294、翻转游戏2
     */
    public boolean canWin(String s) {
        if(s==null||s.length()==0){
            return false;
        }

        return canWinHelper(s.toCharArray());
    }

    public boolean canWinHelper(char[] arr){
        for(int i=0; i<arr.length-1;i++){
            if(arr[i]=='+'&&arr[i+1]=='+'){
                arr[i]='-';
                arr[i+1]='-';

                boolean win = canWinHelper(arr);

                arr[i]='+';
                arr[i+1]='+';

                //if there is a flip which makes the other player lose, the first play wins
                if(!win){
                    return true;
                }
            }
        }

        return false;
    }


    /***
     * 剑指offer--左旋字符串
     */

    public String LeftRotateString(String str,int n){
        if (str == null || str.length() <2 || str.length() <n) return str;
        char[] chars = str.toCharArray();
        //先翻转前n个元素
        reverse(chars,0,n-1);

        //再翻转后面的
        reverse(chars,n,chars.length-1);

        //整体翻转
        reverse(chars,0,chars.length-1);
        return String.valueOf(chars);
    }

    //发散思维，右旋
    public String RightRotateString(String str,int n){
        if (str == null ||str.length() < 2 || str.length() <n) return str;
        char[] arr = str.toCharArray();
        //先整体翻转
        reverse(arr,0,arr.length-1);
        //再翻转前半部分
        reverse(arr,0,n-1);
        //再翻转后半部分
        reverse(arr,n,arr.length-1);
        return String.valueOf(arr);

    }

    /**
     * 剑指offer--翻转单词的顺序
     * */
    public String ReverseSentence(String str){
        if (str == null || str.length() == 0) return str;
        char[] arr = str.toCharArray();
        //翻转整个句子
        reverse(arr,0,arr.length-1);

        for (int i=0;i<arr.length;i++){
            int low=i;
            //遇到空格就跳
            while (i<arr.length && arr[i] != ' ') i++;
            //翻转单个单词
            int high=i-1;
            reverse(arr,low,high);
        }
        return new String(arr);
    }

    /***
     * 中等--3、无重复字符的最长子串(函数原名不带2)
     */
    public int lengthOfLongestSubstring2(String s) {
        if (s == null || s.length() == 0) return 0;
        HashMap<Character, Integer> map = new HashMap<>();
        int startIndex=0,max=0;
        for (int i=0;i<s.length();i++){
            if (map.containsKey(s.charAt(i))){
                int oldIndex = map.get(s.charAt(i));
                startIndex=Math.max(startIndex,oldIndex+1);
            }
            map.put(s.charAt(i),i);//放入指定的字符，以及更行最新的位置
            max=Math.max(max,i-startIndex+1);//更新最大字符串长度
        }
        return max;
    }

    //二刷
    public int lengthOfStrs(String s){
        if (s == null || s.length() == 0) return 0;
        Map<Character, Integer> map = new HashMap<>();
        int leftIx=0,max=0;
        for (int i=0;i<s.length();i++){
            if (map.containsKey(s.charAt(i))){
                leftIx=Math.max(leftIx,map.get(s.charAt(i))+1);
            }
            map.put(s.charAt(i),i);
            max=Math.max(max,i-leftIx+1);
        }
        return max;
    }

    /****
     * 简单--67、二进制求和
     */
    public String addBinary(String a, String b) {
        StringBuilder res=new StringBuilder();
        int p1=a.length()-1,p2=b.length()-1;
        int resn=0;//表示余数
        while(p1>=0 || p2>=0){
            int i1=p1>=0?a.charAt(p1)-'0':0;
            int i2=p2>=0?b.charAt(p2)-'0':0;
            int sum=i1+i2+resn;
            res.append(sum%2);//求余数
            resn=sum/2;
            p1--;
            p2--;
        }
        if (resn == 1)
            res.append(resn);
        return res.reverse().toString();
    }

    /****
     * 困难--76、最小覆盖子串
     */
    public String minWindow(String S, String T) {
        int[] map=new int[128];
        for (int i=0;i<T.length();i++){
            map[T.charAt(i)]++;//记录T中各个字符出现的次数
        }
        int start=0,end=0,d=Integer.MAX_VALUE,counter=T.length(),head=0;
        while(end<S.length()){
            if (map[S.charAt(end++)]-- >0) counter--;//只有目标串中的字符才会成

            while (counter == 0){//说明窗口内的元素都包含了
                if (end - start <d) d=end - (head = start);
                if (map[S.charAt(start++)]++ >= 0) counter++;//如果目标串中的元素被删除了就计数+1
            }
        }
        return d == Integer.MAX_VALUE?"":S.substring(head,head+d);
    }

    /***
     * 简单--796、旋转字符串
     */
    public boolean rotateString(String A, String B) {
        if (A.length() != B.length()) return false;
        return (A+A).contains(B);
    }


    /***
     * 中等--394、字符串解码
     */
    public String decodeString(String s){
        String res="";
        //记录'['之前的数字
        Stack<Integer> countStack = new Stack<>();
        //记录'['之前的运算结果
        Stack<String> resStack = new Stack<>();
        int ix=0;
        int curNum=0;
        while (ix < s.length()){
            char ch = s.charAt(ix);
            if (Character.isDigit(ch)){
                while (Character.isDigit(s.charAt(ix)))
                    curNum=10*curNum+(s.charAt(ix++)-'0');
            }else if (ch == '['){
                resStack.push(res);
                res="";//注意置空
                countStack.push(curNum);
                curNum=0;
                ix++;
            }else if (ch == ']'){

                StringBuilder tmp = new StringBuilder(resStack.pop());
                int repeatTimes=countStack.pop();
                for (int i=0;i<repeatTimes;i++){
                    tmp.append(res);
                }

                res=tmp.toString();
                ix++;

            }else{//就是字母
                res+=s.charAt(ix++);
            }
        }
        return res;
    }
    //二刷
    public String decodeString1_2(String s){
        String res="";
        int curNum=0;
        Stack<Integer> int_Stacks = new Stack<>();
        Stack<String> strs_Stacks = new Stack<>();
        int ix=0;
        while(ix < s.length()){
            char c = s.charAt(ix);

            if (Character.isDigit(c)){

                while(Character.isDigit(s.charAt(ix)))
                    curNum=curNum*10+(s.charAt(ix++)-'0');

            }else if (c == '['){
                int_Stacks.push(curNum);
                curNum=0;
                strs_Stacks.push(res);
                res="";
                ix++;

            }else if (c == ']'){

                StringBuilder tmp = new StringBuilder(strs_Stacks.pop());
                int times = int_Stacks.pop();

                for (int i=0;i<times;i++){
                    tmp.append(res);
                }
                res=tmp.toString();
                ix++;

            }else{//字母情况
                res+=s.charAt(ix++);
            }
        }
        return res;
    }

    /***
     * 中等--3、最长不含重复字符的子字符串
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) return 0;
        Map<Character, Integer> map = new HashMap<>();
        int leftIx=0,max=0;
        for (int i=0;i<s.length();i++){
            if (map.containsKey(s.charAt(i))){
                leftIx=Math.max(leftIx,map.get(s.charAt(i))+1);//更新左边的位置
            }
            map.put(s.charAt(i),i);//更新最新的位置
            max=Math.max(max,i-leftIx+1);
        }
        return max;
    }

    /***
     * 中等--93、复原IP地址
     */
    public List<String> restoreIpAddresses(String s){
        int len = s.length();
        List<String> res = new ArrayList<>();
        if (len < 4 || len > 12) return res;
        //用来存储当前的路径
        Deque<String> path = new ArrayDeque<>(4);
        int splitTimes=0;
        dfs(s,len,splitTimes,0,path,res);
        return res;
    }

    /**
     * 判断 s 的子区间 [left, right] 是否能够成为一个 ip 段
     * 判断的同时顺便把类型转了
     * */
    private int judgeIfIpSegment(String s,int left,int right){
        int len=right-left+1;
        if (len > 1 && s.charAt(left) == '0') return -1;
        //转成int类型
        int res=0;
        for (int i=left;i<= right;i++){
            res= res*10 + s.charAt(i) - '0';
        }

        if (res > 255) return -1;
        return res;
    }

    private void dfs(String s, int len, int splitTimes, int begin, Deque<String> path, List<String> res) {
        if (begin == len){
            if (splitTimes == 4) res.add(String.join(".",path));
            return;
        }
        //看到剩下的不够了，就退出(剪枝)，len-begin表示剩余的还未分割的字符串的位数
        if (len - begin < (4-splitTimes) || len - begin >3*(4-splitTimes)) return;
        for (int i=0;i<3;i++){
            if (begin + i >= len) break;
            int ipSegment = judgeIfIpSegment(s,begin,begin+i);
            if (ipSegment != -1){
                path.addLast(ipSegment+"");
                dfs(s,len,splitTimes+1,begin+i+1,path,res);
                path.removeLast();
            }
        }
    }

    public static void main(String[] args) {
        String s="pwwkew";
        Strs ins = new Strs();
        int i = ins.lengthOfLongestSubstring(s);

    }
}
