
public class shuangxianglianbiao {

        public static void main(String[] args) {
            //创建节点
            Hero2 hero1 = new Hero2(1,"宋江","及时雨");
            Hero2 hero2 = new Hero2(2,"吴用","智多星");
            Hero2 hero3 = new Hero2(3,"林冲","豹子头");
            Hero2 hero4 = new Hero2(4,"武松","行者");
            Doublelist doublelist = new Doublelist();
            //doublelist.add(hero1);
            //doublelist.add(hero2);
            //doublelist.add(hero3);
            //doublelist.add(hero4);
            doublelist.addList(hero1);
            doublelist.addList(hero3);
            doublelist.addList(hero2);
            doublelist.addList(hero4);

            doublelist.list();
            System.out.println("--------------");

            //Hero2 newhero = new Hero2(2,"卢俊义","玉麒麟");
            //doublelist.replaceList(newhero);
            //doublelist.list();
            //System.out.println("--------------");

            //doublelist.deleteList(4);
            //doublelist.list();
        }
    }

    //创建一个双向链表的类
    class Doublelist{
        //先初始化一个头结点，头结点不动,不存放具体的数据
        private Hero2 head =new Hero2(0, "", "");

        //遍历双向链表
        //不考虑编号顺序时
        public  void list() {
            if(head.next == null) {
                System.out.println("该链表为空");
                return;
            }
            //创建一个辅助变量
            Hero2 temp = head.next;
            while(true) {
                if(temp ==null) {
                    break;
                }
                //输出数据
                System.out.println(temp);
                //将temp后移
                temp = temp.next;
            }
        }
        //添加节点到双向链表
        public  void add(Hero2 hero) {
            //因为head节点不能动，需要一个辅助变量temp
            Hero2 temp = head;
            //遍历链表(死循环)
            while(true) {
                //找到链表最后
                if(temp.next == null) {
                    break;
                }
                //如果不是最后，temp后移
                temp =temp.next;

            }
            //当退出while循环时，temp就是指向链表的最后
            temp.next = hero;
            hero.pre = temp;
        }
        //将数据按照排名插入到指定位置
        public void addList(Hero2 hero) {
            Hero2 temp = head;
            if (temp.no>hero.no){//判断是否要插在链表头的情况
                hero.next=head;
                head.pre=hero;
                head=hero;
                return;
            }
            //到了这，说明插入点不在队头
            //判断tmp的下一个节点的序号是否小于待插入节点节点的序号或者下一个节点是否为空
            while(temp.next != null){
                if (temp.next.no<hero.no){
                    temp=temp.next;
                }else{
                    break;
                }
            }
            if (temp.next !=null){//说明还没到末尾
                hero.next=temp.next;//插入过程
                temp.next.pre=hero;
            }
                temp.next=hero;
                hero.pre=temp;
        }

}
class Hero2{
    public int no;
    public String name;
    public String nickname;
    public Hero2 next;//指向下一个节点，默认为null
    public Hero2 pre;//指向前一个节点，默认为null
    //构造器
    public Hero2(int no,String name,String nickname) {
        this.no = no;
        this.name = name;
        this.nickname =nickname;
    }
    //为了显示方法，重新toString方法
    @Override
    public String toString() {
        // TODO Auto-generated method stub
        return "Hero [no=" + no +",name="+ name +",nickname=" + nickname +" ]";
    }
}
