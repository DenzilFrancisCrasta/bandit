
rewards = cell(1,3)
optimality = cell(1,3)

idx = 1;
for i = [0, 0.1, 0.01]
  rewards{1,idx} = load(strcat('oorewards',num2str(i),'.txt'));
  optimality{1,idx} = load(strcat('oooptimality',num2str(i),'.txt'));
  idx = idx+ 1;
end

colors = ['g','k', 'r'];
for i=[1:3]
  plot([1:1000],rewards{i}, colors(i));
  hold on;
end

xlabel('Steps');
ylabel('Average Reward');
axis([-10 1000 0 1.5])

