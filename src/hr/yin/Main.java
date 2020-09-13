package hr.yin;

import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

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

}
