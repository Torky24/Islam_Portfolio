import java.awt.*;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ViewGrid extends JPanel {

    JFrame frame = new JFrame();
    int t;
    int height;
    int width;

    public ViewGrid(int t, int height, int width) { // constructor
        frame.setContentPane(this);
        setLayout(new GridLayout(t, t, 0, 0));
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);

        this.t = t;
        this.width = width;
        this.height = height;

    }

    public void addButtons(int[][] gridCells) {
        JButton grid;
        for (int y = 0; y < t; y++) {
            for (int x = 0; x < t; x++) {
                grid = new JButton(); // creates new button
                grid.setPreferredSize(new Dimension(55, 55));
                if (gridCells[x][y] == 1) {
                    grid.setBackground(Color.RED);
                    add(grid); // adds button to grid
                }
                else{
                    grid.setBackground(Color.GRAY);
                    add(grid); // adds button to grid
                }
            }
        }
    }
}