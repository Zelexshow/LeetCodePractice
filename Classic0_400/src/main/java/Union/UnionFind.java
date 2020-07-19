package Union;

/***
 * 并查集方面的
 */
public class UnionFind {
    int[] parent;
    int[] rank;

    public UnionFind(int total){
        parent=new int[total];
        rank=new int[total];

        for (int i=0;i<total;i++){
            parent[i]=i;
            rank[i]=i;
        }
    }
    public int find(int x){
        while(x != parent[x]){
            parent[x]=parent[parent[x]];
            x=parent[x];
        }
        return x;
    }
    public void UnionElements(int p,int q){
        int pRoot = find(p);
        int qRoot=find(q);

        if (pRoot == qRoot) return;
        if (rank[pRoot]<rank[qRoot]) parent[pRoot]=qRoot;
        else if (rank[pRoot]<rank[qRoot]) parent[qRoot]=pRoot;
        else {
            parent[pRoot] = qRoot;
            rank[qRoot]+=1;
        }
    }
    /***
     * 中等--684、冗余连接
     */
    public int[] findRedundantConnection(int[][] edges) {
        int[] res=new int[2];
        UnionFind unionFind = new UnionFind(edges.length);
        unionFind.UnionElements(edges[0][0]-1,edges[0][1]-1);
        for (int i=1;i<edges.length;i++){
            if (unionFind.find(edges[i][0]-1) == unionFind.find(edges[i][1]-1)){
                res[0]=edges[i][0];
                res[1]=edges[i][1];
            }else unionFind.UnionElements(edges[i][0]-1,edges[i][1]-1);
        }
        return res;
    }

    public static void main(String[] args) {
        int[][] arr={{1,2},{2,3},{3,4},{1,4},{1,5}};
        UnionFind unionFind = new UnionFind(5);
        int[] redundantConnection = unionFind.findRedundantConnection(arr);
        System.out.println(redundantConnection);
    }
}
