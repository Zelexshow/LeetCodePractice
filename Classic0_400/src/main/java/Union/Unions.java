package Union;

import java.util.Arrays;

public class Unions {
    int[] parent;//记录zhen

    int find(int[] parent,int i){
        if (parent[i] == -1) return i;
        return find(parent,parent[i]);
    }
    void union(int[] parent,int x,int y){
        int xset = find(parent, x);
        int yset = find(parent, y);
        if (xset != yset) parent[xset]=yset;
    }
    public boolean validTree(int n,int[][] edges){
        int len=edges.length;
        parent=new int[n];
        Arrays.fill(parent,-1);
        for (int i=0;i<len;i++){
            if (find(parent, edges[i][0]) == find(parent, edges[i][1])) {
                return false;
            }
            union(parent,edges[i][0],edges[i][1]);
        }
        int count=0;
        for (int i=0;i<n;i++){
            if (parent[i] == -1) count++;
        }
        return count == 1;
    }
}

