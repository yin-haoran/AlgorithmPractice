package hr.yin;

import java.util.*;

public class Main {

    public static void main(String[] args) {
    }

    /**
     * 数组中重复数字
     * 0.两重for
     * 1.HashSet(HashMap)判断当前数字是否出现过 (相比于0：空间换时间
     * 2.bucket数组记录每个数字是否出现过 (相比于1：数组代替哈希表
     * 3.数字和数组下标一致 (相比于2：减小空间复杂度
     */
    public int findRepeatNumber(int[] nums) {
        // 从第一个元素开始遍历数组
        for (int i = 0; i < nums.length; i++) {
            // 当前数字与下标不一致时，将数字交换到正确下标
            while (nums[i] != i) {
                // 当前数字待交换去的位置有正确的数字，找到重复元素
                if (nums[nums[i]] == nums[i]) {
                    return nums[i];
                }

                int tmp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = tmp;
            }
        }

        // 没有重复元素
        return -1;
    }

    /**
     * 排序二维数组中查找
     * 左上角为最小值，右下角为最大值
     */
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        // 从右上角元素比较，每次比较可排除一行/列
        int row = 0;
        int column = matrix[0].length - 1;

        while (row < matrix.length && column >= 0) {
            if (matrix[row][column] == target) {
                return true;
            } else if (target > matrix[row][column]) {
                row++;
            } else {
                column--;
            }
        }

        // 整个数组未找到target
        return false;
    }

    /**
     * 替换空格
     * 1.String#replaceAll() 性能不好
     * 2.StringBuilder拼接   本质是：char[]+扩容（System.arraycopy()）
     * 3.定义静态数组：大小刚好（遍历两次字符串，损失时间）、大小为字符串3倍（损失空间）
     */
    public String replaceSpace(String s) {
        if (s == null) {
            return null;
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(s.charAt(i));
            }
        }

        return sb.toString();
    }

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    /**
     * 从尾到头打印单链表
     * 必须从头到尾遍历链表
     * 1.可用栈(Stack extends Vector)存储遍历的每个值（多用一个栈的空间）
     * 2.访问一次链表记录节点数、再次访问链表存储节点值（多了遍历一次的时间）
     */
    public int[] reversePrint(ListNode head) {
        ListNode p = head;
        int count = 0;
        while (p != null) {
            count++;
            p = p.next;
        }

        int[] result = new int[count];
        while (head != null) {
            result[--count] = head.val;
            head = head.next;
        }

        return result;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /**
     * 前序和中序重建二叉树。不含重复值
     * 1.直接在中序遍历中查找头结点位置
     * 2.HashMap存储中序遍历值及其下标（提高了查找速度，但增加了空间的使用
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || preorder.length == 0 || inorder == null || inorder.length == 0) {
            return null;
        }

        // HashMap存储中序遍历元素及其下标，方便查找
        Map<Integer, Integer> inMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inMap.put(inorder[i], i);
        }

        return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, inMap);
    }

    private TreeNode buildTree(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd, Map<Integer, Integer> inMap) {
        // 边界
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }

        // 当前确定的头结点
        TreeNode head = new TreeNode(preorder[preStart]);

        // 头结点在中序的下标
        int headPosInOrder = inMap.get(head.val);
        // 左孩子节点数
        int leftChildNum = headPosInOrder - inStart;
        head.left = buildTree(preorder, preStart + 1, preStart + leftChildNum, inorder, inStart, headPosInOrder - 1, inMap);
        head.right = buildTree(preorder, preStart + leftChildNum + 1, preEnd, inorder, headPosInOrder + 1, inEnd, inMap);
        return head;
    }

    /**
     * 两个栈实现队列
     */
    class CQueue {

        private final Stack<Integer> inputStack;
        private final Stack<Integer> outputStack;

        public CQueue() {
            inputStack = new Stack<>();
            outputStack = new Stack<>();
        }

        public void appendTail(int value) {
            inputStack.push(value);
        }

        public int deleteHead() {
            if (!outputStack.isEmpty()) {
                return outputStack.pop();
            }

            if (inputStack.isEmpty()) {
                return -1;
            } else {
                while (!inputStack.isEmpty()) {
                    outputStack.push(inputStack.pop());
                }
                return outputStack.pop();
            }
        }

    }

    /**
     * 斐波那契数列
     * 0 1 1 2 3 5 ...
     */
    public int fib(int n) {
        if (n == 0 || n == 1) {
            return n;
        }

        int firstNumber = 0;
        int secondNumber = 1;
        int tmp;
        while (n >= 2) {
            tmp = (firstNumber + secondNumber) % 1000000007;
            firstNumber = secondNumber;
            secondNumber = tmp;
            --n;
        }
        return secondNumber;
    }

    /**
     * 青蛙跳台阶
     * f(n) = f(n-1) + f(n-2)
     * f(0) = 1   f(1) = 1   f(2) = 2   f(3) = 3
     */
    public int numWays(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }

        int firstNum = 1;
        int secondNum = 1;
        int tmp;
        while (n >= 2) {
            tmp = (firstNum + secondNum) % 1000000007;
            firstNum = secondNum;
            secondNum = tmp;
            --n;
        }

        return secondNum;
    }

    /*
        二分查找：
            左、中、右。（以中间值作为基准,进行大、小、相等的比较）
            每次查找会排除掉一些元素，且数组元素特性保持不变
            注意分析数组元素个数为1、2的情况
     */
    /**
     * 旋转数组的最小值
     *
     * 数组情况：没旋转---递增序列(最小值在左边)    旋转了---最小值靠左边、最小值靠右边
     * 特殊值：第一个元素、最后个元素
     *
     * 中间值和边界值相等->不能判断最小值在左边或右边->直接忽略边界值
     *
     * 不能分类的情况，可以用特殊条件来过滤。
     * 以第一个元素作为待比较值，当numbers[mid] > numbers[left]时：1.没旋转，最小值在左边，为第一个元素。2.旋转了，最小值在右边
     * 此时可以判断：如果numbers[left] < number[right]，那么为1，直接返回最小值numbers[left]
     */
    public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        int left = 0;
        int right = numbers.length - 1;
        int mid;
        while (left < right) {
            mid = (left + right) / 2; // mid = left + (right - left) / 2;
            if (numbers[mid] > numbers[right]) {
                left = mid + 1;
            } else if (numbers[mid] < numbers[right]) {
                right = mid;
            } else {
                right = right - 1;
            }
        }

        return numbers[left];
    }
    /*public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }

        int left = 0;
        int right = numbers.length - 1;
        int mid;
        while (right - left > 1) {
            mid = (left + right) / 2;
            if (numbers[mid] > numbers[left]) {
                if(numbers[left] < numbers[right]) {
                    return numbers[left];
                }
                left = mid + 1;
            } else if (numbers[mid] < numbers[left]) {
                right = mid;
            } else {
                left =left + 1;
            }
        }
        return numbers[left] > numbers[right] ? numbers[right] : numbers[left];
    }*/

    /**
     * 矩阵中的路径
     *
     * DFS 回溯 递归
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0] == null || board[0].length == 0) {
            return false;
        }
        if (word == null || word.length() == 0) {
            return false;
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (checkDFS(board, i, j, word, 0)) {
                    return true;
                }
            }
        }

        return false;
    }
    /**
     * @param board 矩阵
     * @param i 当前行
     * @param j 当前列
     * @param word 路径
     * @param index 当前要比较字符的下标
     */
    private boolean checkDFS(char[][] board, int i, int j, String word, int index) {
        if (index == word.length()) {
            return true;
        }

        if (i < 0 || i >= board.length ||
            j < 0 || j >= board[0].length ||
            board[i][j] != word.charAt(index)) {
            return false;
        }

        board[i][j] = '\0';
        boolean result =
            checkDFS(board, i + 1, j, word, index + 1) ||
            checkDFS(board, i - 1, j, word, index + 1) ||
            checkDFS(board, i, j + 1, word, index + 1) ||
            checkDFS(board, i, j - 1, word, index + 1);
        board[i][j] = word.charAt(index);

        return result;
    }

    /**
     * 机器人的运动范围
     *
     * DFS 递归
     * BFS 队列
     */
    public int movingCount(int m, int n, int k) {
        if (m <= 0 || n <= 0 || k < 0) {
            return 0;
        }

        boolean[][] visited = new boolean[m][n];
        return movingDFS(0, 0, k, visited);
    }
    /**
     *
     * @param i 当前行号
     * @param j 当前列号
     * @param k 下标数位之和的限制
     * @param visited 标记被访问过的位置
     * @return 当前位置继续移动，总共可以移动的位置数量
     */
    private int movingDFS(int i, int j, int k, boolean[][] visited) {
        if (i >= visited.length || i < 0 ||
            j >= visited[0].length || j < 0 ||
            visited[i][j] || getNumSum(i) + getNumSum(j) > k) {
            return 0;
        }

        visited[i][j] = true;
        return 1 +
            movingDFS(i + 1, j, k, visited) +
            movingDFS(i, j + 1, k, visited);
    }
    /** 获取一个数的数位之和 */
    private int getNumSum(int num) {
        int sum = 0;

        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }

        return sum;
    }

}
