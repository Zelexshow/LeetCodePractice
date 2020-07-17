package Stack;

import java.util.*;

/***
 *
 * 栈相关
 *
 */
public class Stacks {
    /***
     * 简单--20、有效括号
     */
    public boolean isValid(String s){
        Map<Character,Character> map=new HashMap<>();
        map.put(')','(');
        map.put(']','[');
        map.put('}','{');
        Stack<Character> stack = new Stack<>();
        for (int i=0;i<s.length();i++){
            char c = s.charAt(i);
            if (map.containsKey(c)){
                char topElement=stack.empty()?'#': stack.pop();
                if (topElement != map.get(c)) return false;
            }else{
                stack.push(c);//开括号就压栈
            }
        }
        return stack.isEmpty();
    }
    //二刷
    public boolean isValid1_2(String s){
        HashMap<Character, Character> map = new HashMap<>();
        map.put(')','(');
        map.put(']','[');
        map.put('}','{');
        Stack<Character> stack = new Stack<>();
        for (int i=0;i<s.length();i++){
            char c=s.charAt(i);
            if (map.containsKey(c)){
                char topElement = stack.isEmpty()?'#':stack.pop();
                if (topElement != map.get(c)) return false;
            }else{
                stack.push(s.charAt(i));//左括号就直接push
            }
        }
        return true;
    }

    /***
     * 困难--32、最长有效括号
     */
    public int longestValidParentheses(String s) {
        int max=0;
        Stack<Integer> stack = new Stack();
        stack.push(-1);
        for (int i=0;i<s.length();i++){
            if (s.charAt(i) == '('){
                stack.push(i);
            }else{
                stack.pop();
                if (stack.empty()){
                    stack.push(i);
                }else{
                    max=Math.max(max,i-stack.peek());
                }
            }
        }
        return max;
    }
    //二刷
    public int longestValidParenthess1_2(String s){
        if (s == null || s.length() == 0) return 0;
        int max=0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i=0;i<s.length();i++){
            if (s.charAt(i) == '(') stack.push(i);
            else{
                stack.pop();
                if (stack.empty()){
                    stack.push(i);
                }else{
                    max=Math.max(max,i-stack.peek());
                }
            }
        }
        return max;
    }
    /***
     * 中等--71、简化路径
     */
    public String simplifyPath(String path){
        String[] s = path.split("/");
        Stack<String> stack = new Stack<>();

        for (int i=0;i<s.length;i++){
            if (!stack.isEmpty() && s[i].equals("..")){
                stack.pop();
            }else if (!s[i].equals("") && !s[i].equals(".")
            && !s[i].equals("..")){
                stack.push(s[i]);
            }
        }
        if (stack.isEmpty())
            return "/";
        StringBuffer res = new StringBuffer();
        for (int i=0;i<stack.size();i++){
            res.append("/"+stack.get(i));
        }
        return res.toString();
    }
    /***
     * 中等--150、逆波兰表达式
     */
    //使用栈
    public int evalRPN(String[] tokens) {
        Stack<Integer> numStack = new Stack<>();
        Integer op1,op2;
        for (String s:tokens){
            switch (s){
                case "+":
                    op2 = numStack.pop();
                    op1 = numStack.pop();
                    numStack.push(op1+op2);
                    break;
                case "-":
                    op2 = numStack.pop();
                    op1 = numStack.pop();
                    numStack.push(op1-op2);
                    break;
                case "*":
                    op2 = numStack.pop();
                    op1 = numStack.pop();
                    numStack.push(op1*op2);
                    break;
                case "/":
                    op2 = numStack.pop();
                    op1 = numStack.pop();
                    numStack.push(op1/op2);
                    break;
                default:
                    numStack.push(Integer.valueOf(s));
                    break;
            }
        }
        return numStack.pop();
    }

    /***
     * 简单--155、最小栈
     */
    class MiniStack{
        private Stack<Integer> dataStack;
        private Stack<Integer> minStack;

        public MiniStack(){
            dataStack=new Stack<>();
            minStack=new Stack<>();
        }
        public void push(int x){
            dataStack.push(x);
            if (minStack.isEmpty()){
                minStack.push(x);
            }else if (x <= getMin()){
                minStack.push(x);
            }else{
                minStack.push(getMin());
            }
        }
        public void pop(){
            if (dataStack.isEmpty()){
                throw new RuntimeException("Stacki ie empty,can not pop");
            }else{
                dataStack.pop();
                minStack.pop();
            }
        }
        public int top(){
            return dataStack.peek();
        }

        public int getMin(){
            return minStack.peek();
        }
    }

    /***
     * 困难--224、基本计算器：只有加减号
     */
    //仅含有加法、减法以及小括号
    /**双栈法*/
    public int calculate(String s) {
        char[] array = s.toCharArray();
        int n = array.length;
        Stack<Integer> num = new Stack<>();
        Stack<Character> ops = new Stack<>();

        int tmp=-1;
        for (int i=0;i<n;i++){
            if (array[i] == ' ') continue;//如果是空格
            //数字进行累加
            if(isNumber(array[i])){//如果是数字
                if (tmp == -1) tmp=array[i]-'0';
                else tmp=tmp*10+array[i]-'0';
            }else{//如果是运算符，首先将整个数字入栈
                if (tmp != -1){
                    num.push(tmp);
                    tmp=-1;
                }
                if (isOperation(array[i]+"")){
                    while(!ops.isEmpty()){
                        if (ops.peek() == '(') break;
                        //不经出站，进行运算，并将结果压入栈中
                        int num1 = num.pop();
                        int num2 = num.pop();
                        if (ops.pop() == '+') num.push(num1+num2);
                        else num.push(num2-num1);
                    }
                    //将当前运算符压入栈中
                    ops.push(array[i]);
                }else{
                    //遇到左括号直接入栈
                    if (array[i] == '('){
                        while(ops.peek() != '('){
                            int num1 = num.pop();
                            int num2 = num.pop();
                            if (ops.pop() == '+'){
                                num.push(num1+num2);
                            }else num.push(num2-num1);
                        }
                        ops.pop();
                    }
                }
            }
        }
        if (tmp != -1){
            num.push(tmp);
        }
        //将栈中其他元素继续元素
        while(!ops.isEmpty()){
            int num1=num.pop();
            int num2=num.pop();
            if (ops.pop() == '+'){
                num.push(num1+num2);
            }else num.push(num2 - num1);
        }
        return num.pop();
    }
    private boolean isNumber(char c){
        return c>= '0' && c <= '9';
    }
    private boolean isOperation(String t){
        return t.equals("+") || t.equals("-") || t.equals("*")
                || t.equals("/");
    }

    /***
     * 中等--基本计算器2
     */
    public int caculate2(String s){
        Stack<Integer> numStack = new Stack<>();
        char lastOp='+';
        char[] arr = s.toCharArray();
        for (int i=0;i<arr.length;i++){
            if (arr[i] == ' ') continue;
            if (Character.isDigit(arr[i])){
                int tmpNum=arr[i]-'0';
                while (++i <arr.length && Character.isDigit(arr[i])){
                    tmpNum=tmpNum*10+(arr[i]-'0');
                }
                i--;
                if (lastOp == '+') numStack.push(tmpNum);
                else if (lastOp == '-') numStack.push(-tmpNum);
                else numStack.push(res(lastOp,numStack.pop(),tmpNum));
            }else lastOp =arr[i];
        }
        int ans=0;
        for (int num:numStack) ans+=num;
        return ans;
    }

    private int res(char lastOp, int a, int b) {
        if (lastOp == '*') return a*b;
        else if (lastOp == '/') return a/b;
        else if (lastOp == '+') return a+b;
        else return a-b;
    }

    /***
     * 简单--225、用队列实现栈
     */
    public class MyStack{
        private Queue<Integer> queue;//用于压栈和从栈顶取元素
        private Queue<Integer> help;//用于协助取元素的队列
        private int top;//表示栈顶元素

        public MyStack(){
            queue=new LinkedList<>();
            help=new LinkedList<>();
        }
        public void push(int x){
            queue.add(x);
            top=x;//表示栈顶元素
        }
        public int pop(){
            //相当于使用辅助队列来进行调换
            if (queue.isEmpty()){
                throw new RuntimeException("栈为空！");
            }
            while(queue.size()!=1){//取到还剩一个元素
                top=queue.poll();
                help.add(top);
            }
            int res=queue.poll();//最后一个对应栈中最后面的数
            Queue<Integer> temp=queue;
            queue=help;
            help=temp;
            return res;
        }
        public int top(){
            if (queue.isEmpty()) throw new RuntimeException("栈为空！");
            return top;
        }
        public boolean empty(){
            return queue.isEmpty();
        }
    }
    /***
     * 简单--232、用栈实现队列
     * 这个体型都要求两个容器交替使用，交替倒腾，实现取数
     */
    public class MyQueue{
        private Stack<Integer> pushStack;//用于压栈的
        private Stack<Integer> popStack;//辅助使用
        public MyQueue(){
            pushStack=new Stack<>();
            popStack=new Stack<>();
        }
        public void push(int x){
            if (popStack.isEmpty()){
                while(!pushStack.isEmpty()){
                    popStack.push(pushStack.pop());
                }
            }
            pushStack.push(x);
        }
        public int pop(){
            if (popStack.isEmpty() && pushStack.isEmpty()) throw new RuntimeException("栈为空！");
            if (popStack.isEmpty()){
                while(!pushStack.isEmpty()){
                    popStack.push(pushStack.pop());
                }
            }
            return popStack.pop();
        }
        public int peek(){
            if (popStack.isEmpty() && pushStack.isEmpty()) throw new RuntimeException("栈为空！");
            if (popStack.isEmpty()){
                while(!pushStack.isEmpty()){
                    popStack.push(pushStack.pop());
                }
            }
            return popStack.peek();
        }
        public boolean empty(){
            return popStack.isEmpty() && pushStack.isEmpty();
        }
    }
    /***
     * 中等--739、每日温度
     * 通过压栈的方式，将栈顶的元素（最近的索引）和当前的索引的元素进行比较
     * 本质上是通过弹栈来更新返回的结果列表
     */
    public int[] dailyTemperatures(int[] T) {
        int len = T.length;
        int[] dist=new int[len];//返回的结果值
        Stack<Integer> indexs = new Stack<>();

        for (int curIndex=0;curIndex<len;curIndex++){
            while(!indexs.isEmpty() && T[curIndex]> T[indexs.peek()]){
                int preIndex=indexs.pop();
                dist[preIndex]=curIndex-preIndex;
            }
            indexs.add(curIndex);//相当于压到栈顶,add和push对于栈来说，均是压入到栈顶
        }
        return dist;
    }

    /***
     * 中等--503、下一个更大的元素2
     */
    public int[] nextGreaterElements(int[] nums) {
        int n=nums.length;
        int[] next=new int[n];
        Arrays.fill(next,-1);
        Stack<Integer> stack = new Stack<>();
        for (int i=0;i<n*2;i++){
            int num=nums[i%n];
            while(!stack.isEmpty() && num>nums[stack.peek()]){
                next[stack.pop()]=num;
            }
            if (i<n){
                stack.push(i);
            }
        }
        return next;
    }

    /***
     *
     */
    public static void main(String[] args) {
        Stacks ins = new Stacks();
        int[] arr={73, 74, 75, 71, 69, 72, 76, 73};
        int[] res = ins.dailyTemperatures(arr);
        /*Stack<Integer> stack = new Stack();
        stack.push(1);
        stack.push(2);
        stack.add(3);
        System.out.println(stack);*/
    }
}
